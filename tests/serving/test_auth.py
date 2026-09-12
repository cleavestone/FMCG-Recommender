def test_login_with_correct_credentials(client):
    response = client.post("/auth/token", data={"username": "demo", "password": "demo-password"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"] and body["refresh_token"]


def test_login_with_wrong_password(client):
    response = client.post("/auth/token", data={"username": "demo", "password": "wrong"})
    assert response.status_code == 401


def test_login_with_unknown_username(client):
    response = client.post("/auth/token", data={"username": "nobody", "password": "x"})
    assert response.status_code == 401


def test_me_requires_a_valid_token(client):
    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_me_with_valid_token(client, auth_headers):
    response = client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["username"] == "demo"


def test_refresh_token_issues_a_new_access_token(client):
    login = client.post("/auth/token", data={"username": "demo", "password": "demo-password"})
    refresh_token = login.json()["refresh_token"]

    response = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == 200
    new_access_token = response.json()["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {new_access_token}"})
    assert me.status_code == 200


def test_refresh_rejects_an_access_token(client, auth_headers):
    access_token = auth_headers["Authorization"].removeprefix("Bearer ")
    response = client.post("/auth/refresh", json={"refresh_token": access_token})
    assert response.status_code == 401


def test_register_requires_the_admin_key(client):
    response = client.post(
        "/auth/register", json={"username": "new", "password": "pw"}, headers={"X-Admin-Key": "wrong"}
    )
    assert response.status_code == 403


def test_register_and_login_with_new_user(client):
    from fmcg_reco.config import settings

    response = client.post(
        "/auth/register", json={"username": "new", "password": "pw"},
        headers={"X-Admin-Key": settings.admin_api_key},
    )
    assert response.status_code == 201

    login = client.post("/auth/token", data={"username": "new", "password": "pw"})
    assert login.status_code == 200


def test_register_rejects_duplicate_username(client):
    from fmcg_reco.config import settings

    headers = {"X-Admin-Key": settings.admin_api_key}
    client.post("/auth/register", json={"username": "dup", "password": "pw"}, headers=headers)
    response = client.post("/auth/register", json={"username": "dup", "password": "pw"}, headers=headers)
    assert response.status_code == 409
