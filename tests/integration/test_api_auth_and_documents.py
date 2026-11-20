import pytest

pytestmark = pytest.mark.db


async def test_requests_without_a_key_are_rejected(api) -> None:
    response = await api.client.get("/v1/documents")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_api_key"


async def test_requests_with_an_unknown_key_are_rejected(api) -> None:
    response = await api.client.get("/v1/documents", headers={"X-API-Key": "grk_not_a_real_key"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_a_revoked_key_stops_working(api) -> None:
    headers = await api.create_key("temp", {"legal"})
    assert (await api.client.get("/v1/documents", headers=headers)).status_code == 200

    await api.container.api_keys.revoke(1)

    assert (await api.client.get("/v1/documents", headers=headers)).status_code == 401


async def test_only_documents_the_roles_can_read_are_listed(api) -> None:
    legal = await api.create_key("legal-analyst", {"legal"})
    finance = await api.create_key("finance-analyst", {"finance"})
    await api.ingest("Supply Agreement", "Governing law is Delaware.", {"legal"})
    await api.ingest("Revenue Memo", "Revenue share is twelve percent.", {"finance"})

    legal_titles = [
        d["title"]
        for d in (await api.client.get("/v1/documents", headers=legal)).json()["documents"]
    ]
    finance_titles = [
        d["title"]
        for d in (await api.client.get("/v1/documents", headers=finance)).json()["documents"]
    ]

    assert legal_titles == ["Supply Agreement"]
    assert finance_titles == ["Revenue Memo"]


async def test_listing_supports_paging(api) -> None:
    headers = await api.create_key("admin", {"legal"})
    for n in range(3):
        await api.ingest(f"Doc {n}", f"Distinct content number {n}.", {"legal"})

    page = (await api.client.get("/v1/documents?limit=2&offset=2", headers=headers)).json()

    assert len(page["documents"]) == 1
    assert (page["limit"], page["offset"]) == (2, 2)


async def test_the_listing_never_exposes_roles_or_file_paths(api) -> None:
    headers = await api.create_key("analyst", {"legal"})
    await api.ingest("Contract", "Some content.", {"legal", "secret-role"})

    body = (await api.client.get("/v1/documents", headers=headers)).text

    assert "secret-role" not in body
    assert "Contract.txt" not in body


async def test_invalid_paging_parameters_are_rejected_with_the_standard_error(api) -> None:
    headers = await api.create_key("analyst", {"legal"})

    response = await api.client.get("/v1/documents?limit=0", headers=headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_every_response_carries_a_request_id(api) -> None:
    response = await api.client.get("/v1/documents")

    assert response.headers["x-request-id"] == response.json()["error"]["request_id"]
