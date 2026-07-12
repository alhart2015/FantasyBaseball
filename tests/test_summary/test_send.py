import fantasy_baseball.summary.send as send_mod
from fantasy_baseball.summary.send import send_email


def test_send_email_builds_payload_and_returns_id(monkeypatch):
    captured = {}

    def fake_send(payload):
        captured.update(payload)
        return {"id": "msg_123"}

    monkeypatch.setattr(send_mod.resend.Emails, "send", staticmethod(fake_send))

    msg_id = send_email(
        api_key="key_test",
        from_address="digest@x.com",
        recipients=["me@x.com"],
        subject="Subj",
        html="<html></html>",
        text="Subj",
    )
    assert msg_id == "msg_123"
    assert captured["from"] == "digest@x.com"
    assert captured["to"] == ["me@x.com"]
    assert captured["subject"] == "Subj"
    assert captured["html"] == "<html></html>"


def test_send_email_raises_when_no_id(monkeypatch):
    monkeypatch.setattr(send_mod.resend.Emails, "send", staticmethod(lambda payload: {}))
    try:
        send_email(
            api_key="k",
            from_address="a@x.com",
            recipients=["b@x.com"],
            subject="s",
            html="h",
            text="t",
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
