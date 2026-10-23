"""
Airflow DAG for the daily identity-resolution batch.

  * one task per pipeline step; tasks are idempotent and write to data/run=<ds>/
  * retries with exponential backoff for transient failures; data-quality failures raise
    DataQualityError and are marked non-retryable (AirflowFailException) so bad data stops the run
  * catchup=True + start_date allows backfills:  airflow dags backfill -s 2026-10-01 -e 2026-10-07 identity_resolution
  * a run timeout (dagrun_timeout; Airflow 3 removed task SLAs) and a failure callback
    (wire it to Slack / PagerDuty in deployment)
  * max_active_runs=1 so two runs never publish at the same time

In AWS the heavy steps (block, score) would run as EMR Serverless Spark jobs (see spark/ and
infra/terraform/); here every step calls the same local Python functions used in tests.
"""
from datetime import datetime, timedelta

try:                                                    # Airflow 3.x
    from airflow.providers.standard.operators.python import PythonOperator
    from airflow.sdk import DAG
except ImportError:                                     # Airflow 2.x
    from airflow import DAG
    from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException

from irl import pipeline
from irl.quality import DataQualityError


def _notify_failure(context):
    ti = context["task_instance"]
    print(f"ALERT identity_resolution {ti.task_id} failed for {context['ds']} (try {ti.try_number})")


def _call(fn):
    def run(ds, **_):
        try:
            return fn(ds)
        except DataQualityError as e:
            raise AirflowFailException(f"data quality: {e}") from e     # do not retry bad data
    run.__name__ = fn.__name__
    return run


default_args = {
    "owner": "ml-platform",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=20),
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": _notify_failure,
}

with DAG(
    dag_id="identity_resolution",
    description="Standardise, block, match and cluster customer records into identities",
    start_date=datetime(2026, 10, 1),
    schedule="0 4 * * *",
    catchup=True,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    default_args=default_args,
    tags=["identity", "mlops"],
) as dag:
    tasks = [PythonOperator(task_id=name, python_callable=_call(fn)) for name, fn in pipeline.TASKS]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
