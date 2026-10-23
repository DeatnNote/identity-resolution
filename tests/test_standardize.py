from irl.standardize import clean_address, clean_email, clean_phone, split_name, NICK


def test_email_normalisation():
    assert clean_email("J.Smith+promo@GMail.com") == "jsmith@gmail.com"
    assert clean_email("j.smith@yahoo.com") == "j.smith@yahoo.com"      # dots matter outside gmail
    assert clean_email("not-an-email") is None


def test_phone_normalisation():
    assert clean_phone("(212) 555-0101") == "2125550101"
    assert clean_phone("+1 212 555 0101") == "2125550101"
    assert clean_phone("555-0101") is None


def test_address_and_names():
    assert clean_address("12 Main Street, Apt 4") == "12 main st apt 4"
    assert split_name("SMITH") == (None, "smith")
    assert split_name("Bob  Smith") == ("bob", "smith")
    assert NICK["bob"] == "robert"
