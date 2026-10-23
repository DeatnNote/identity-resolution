"""
PySpark blocking + comparison job, and a demonstration of key skew.

Blocking joins records to themselves on a key. With Zipf-distributed surnames and zips a few keys
are huge, and a self-join's cost grows with the SQUARE of block size, so one task gets most of the
work. Three strategies are timed on the same data:

  naive     self-join on the raw key (zip + first 3 letters of surname)
  capped    drop keys larger than MAX_BLOCK before joining (what production blocking does; the
            dropped pairs must be caught by other rules - measured as a recall cost)
  salted    split each oversized block into S sub-blocks and join each sub-block against the
            whole block (all pairs kept, work spread over S tasks)

  python spark/blocking_job.py --people 20000            # local[*]
  spark-submit spark/blocking_job.py --input s3://...     # EMR Serverless (see infra/terraform)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession          # noqa: E402
from pyspark.sql import functions as F                 # noqa: E402

MAX_BLOCK = 100
SALTS = 8


def spark_session():
    return (SparkSession.builder.appName("identity-blocking")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.adaptive.enabled", "false")      # so the skew is visible, not auto-fixed
            .config("spark.sql.autoBroadcastJoinThreshold", "-1")  # force a shuffle join, as at scale
            .getOrCreate())


def load(spark, args):
    if args.input:
        return spark.read.parquet(args.input)
    from irl.generate import generate
    from irl.standardize import standardize
    d = standardize(generate(args.people, 0)[0])
    cols = ["record_id", "person_id", "zip5", "last_name", "first_name", "email_clean", "phone_clean"]
    return spark.createDataFrame(d[cols].astype(object).where(d[cols].notna(), None))


def with_key(df):
    return (df.withColumn("idx", F.monotonically_increasing_id())
              .withColumn("key", F.concat_ws("|", "zip5", F.substring("last_name", 1, 3)))
              .filter(F.col("zip5").isNotNull() & F.col("last_name").isNotNull()))


def pairs_naive(df):
    l, r = df.alias("l"), df.alias("r")
    return (l.join(r, F.col("l.key") == F.col("r.key"))
             .filter(F.col("l.idx") < F.col("r.idx"))
             .select(F.col("l.idx").alias("i_l"), F.col("r.idx").alias("i_r"),
                     F.col("l.person_id").alias("p_l"), F.col("r.person_id").alias("p_r")))


def pairs_capped(df):
    size = df.groupBy("key").count()
    small = df.join(size.filter(F.col("count") <= MAX_BLOCK).select("key"), "key")
    return pairs_naive(small)


def pairs_salted(df):
    """Left side gets a random salt in [0, S); right side is replicated S times. Joining on
    (key, salt) gives every pair exactly once but spreads a big block over S tasks."""
    size = df.groupBy("key").agg(F.count("*").alias("n"))
    d = df.join(size, "key")
    big = F.col("n") > MAX_BLOCK
    left = d.withColumn("salt", F.when(big, (F.rand(7) * SALTS).cast("int")).otherwise(F.lit(0)))
    right = d.withColumn("salt", F.explode(F.when(big, F.array(*[F.lit(i) for i in range(SALTS)]))
                                           .otherwise(F.array(F.lit(0)))))
    l, r = left.alias("l"), right.alias("r")
    return (l.join(r, (F.col("l.key") == F.col("r.key")) & (F.col("l.salt") == F.col("r.salt")))
             .filter(F.col("l.idx") < F.col("r.idx"))
             .select(F.col("l.idx").alias("i_l"), F.col("r.idx").alias("i_r"),
                     F.col("l.person_id").alias("p_l"), F.col("r.person_id").alias("p_r")))


def task_skew(pairs_df):
    """Rows produced per shuffle partition after the join (a proxy for per-task work)."""
    counts = pairs_df.groupBy(F.spark_partition_id().alias("p")).count().toPandas()["count"]
    return {"max_task_rows": int(counts.max()), "median_task_rows": float(counts.median()),
            "max_over_median": round(float(counts.max() / max(counts.median(), 1)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=20000)
    ap.add_argument("--input")
    ap.add_argument("--out", default="results/spark_skew.json")
    args = ap.parse_args()
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    df = with_key(load(spark, args)).cache()
    n = df.count()
    sizes = df.groupBy("key").count().toPandas()["count"]
    true_pairs = (df.groupBy("person_id").count()
                    .select(F.sum(F.col("count") * (F.col("count") - 1) / 2)).first()[0])
    res = {"records": n, "keys": int(len(sizes)), "largest_block": int(sizes.max()),
           "blocks_over_cap": int((sizes > MAX_BLOCK).sum()), "max_block_cap": MAX_BLOCK, "salts": SALTS,
           "true_pairs": int(true_pairs), "strategies": {}}
    for name, fn in (("naive", pairs_naive), ("capped", pairs_capped), ("salted", pairs_salted)):
        p = fn(df)
        t0 = time.time()
        agg = p.agg(F.count("*").alias("pairs"), F.sum((F.col("p_l") == F.col("p_r")).cast("int")).alias("tp")).first()
        secs = time.time() - t0
        res["strategies"][name] = {"candidate_pairs": int(agg["pairs"]), "true_pairs_found": int(agg["tp"]),
                                   "pair_completeness": round(agg["tp"] / true_pairs, 4),
                                   "seconds": round(secs, 2), **task_skew(p)}
        print(name, res["strategies"][name], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    spark.stop()


if __name__ == "__main__":
    main()
