def test_replenishment_requires_auth(client):
    assert client.get("/households/1/replenishment").status_code == 401


def test_replenishment_returns_enriched_items(client, auth_headers):
    response = client.get("/households/1/replenishment", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["household_id"] == 1
    assert body["as_of_day"] == 100
    assert [item["product_id"] for item in body["items"]] == [10, 20]
    assert body["items"][0]["days_until_due"] == 3.0
    assert body["items"][0]["commodity_desc"] == "A"


def test_replenishment_respects_k(client, auth_headers):
    response = client.get("/households/1/replenishment?k=1", headers=auth_headers)
    assert len(response.json()["items"]) == 1
