def test_recommendations_require_auth(client):
    assert client.get("/recommendations/1").status_code == 401


def test_recommendations_returns_enriched_items(client, auth_headers):
    response = client.get("/recommendations/1", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["household_id"] == 1
    assert [item["product_id"] for item in body["items"]] == [10, 20, 30]
    assert body["items"][0]["commodity_desc"] == "A"


def test_recommendations_respects_k(client, auth_headers):
    response = client.get("/recommendations/1?k=2", headers=auth_headers)
    assert [item["product_id"] for item in response.json()["items"]] == [10, 20]


def test_complete_basket_includes_explanation(client, auth_headers):
    response = client.get("/recommendations/1/complete-basket", headers=auth_headers)
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["product_id"] == 40
    assert item["explanation"][0]["antecedent_product_ids"] == [10]
    assert item["explanation"][0]["lift"] == 2.0


def test_similar_items_requires_auth(client):
    assert client.get("/items/10/similar").status_code == 401


def test_similar_items_for_known_product(client, auth_headers):
    response = client.get("/items/10/similar", headers=auth_headers)
    assert response.status_code == 200
    assert [item["product_id"] for item in response.json()["items"]] == [20, 30]


def test_similar_items_for_unknown_product_returns_404(client, auth_headers):
    response = client.get("/items/999999/similar", headers=auth_headers)
    assert response.status_code == 404
