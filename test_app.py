"""Run: python -m unittest test_app.py  (Bedrock is mocked, no AWS calls)."""
import os
import tempfile
import unittest
from unittest.mock import patch

import app as chat_app


def fake_reply(text):
    return {"output": {"message": {"content": [{"text": text}]}}}


class ChatAppTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        chat_app.app.config["DATABASE"] = self.db_path
        self.client = chat_app.app.test_client()
        r = self.client.post("/register", data={"email": "pedro@example.com", "password": "password123"})
        self.assertEqual(r.status_code, 302)

    def tearDown(self):
        os.remove(self.db_path)

    @patch.object(chat_app.bedrock, "converse")
    def test_conversation_is_saved_and_continued(self, converse):
        converse.return_value = fake_reply("Hi Pedro")
        r = self.client.post("/chat", json={"text": "My name is Pedro"})
        self.assertEqual(r.status_code, 200)
        cid = r.json["conversation_id"]
        self.assertEqual(r.json["title"], "My name is Pedro")

        converse.return_value = fake_reply("Pedro")
        r = self.client.post("/chat", json={"conversation_id": cid, "text": "What is my name?"})
        self.assertEqual(r.status_code, 200)
        sent = converse.call_args.kwargs["messages"]
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "user"])

        conv = self.client.get(f"/api/conversations/{cid}").json
        self.assertEqual(len(conv["messages"]), 4)
        self.assertEqual(len(self.client.get("/api/conversations").json), 1)

    @patch.object(chat_app.bedrock, "converse")
    def test_failed_call_saves_nothing(self, converse):
        converse.return_value = fake_reply("   ")
        r = self.client.post("/chat", json={"text": "hello"})
        self.assertEqual(r.status_code, 502)
        self.assertEqual(self.client.get("/api/conversations").json, [])

    @patch.object(chat_app.bedrock, "converse")
    def test_context_window_starts_with_user(self, converse):
        converse.return_value = fake_reply("ok")
        cid = self.client.post("/chat", json={"text": "first"}).json["conversation_id"]
        for i in range(chat_app.MAX_CONTEXT_MESSAGES):
            self.client.post("/chat", json={"conversation_id": cid, "text": f"msg {i}"})
        sent = converse.call_args.kwargs["messages"]
        self.assertEqual(sent[0]["role"], "user")
        self.assertLessEqual(len(sent), chat_app.MAX_CONTEXT_MESSAGES)

    @patch.object(chat_app.bedrock, "converse")
    def test_context_window_char_budget(self, converse):
        converse.return_value = fake_reply("ok")
        cid = self.client.post("/chat", json={"text": "a" * 5000}).json["conversation_id"]
        for i in range(6):
            self.client.post("/chat", json={"conversation_id": cid, "text": str(i) * 5000})
        sent = converse.call_args.kwargs["messages"]
        total = sum(len(m["content"][0]["text"]) for m in sent)
        self.assertLessEqual(total, chat_app.MAX_CONTEXT_CHARS)
        self.assertEqual(sent[0]["role"], "user")
        self.assertEqual(sent[-1]["content"][0]["text"], "5" * 5000)
        self.assertLess(len(sent), 13)  # older turns were dropped

    def test_validation_and_delete(self):
        self.assertEqual(self.client.post("/chat", json={"text": ""}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"text": "x" * 8001}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"conversation_id": 999, "text": "hi"}).status_code, 404)
        self.assertEqual(self.client.post("/chat", json={"conversation_id": "1", "text": "hi"}).status_code, 404)
        self.assertEqual(self.client.delete("/api/conversations/999").status_code, 404)

        with patch.object(chat_app.bedrock, "converse", return_value=fake_reply("ok")):
            cid = self.client.post("/chat", json={"text": "hi"}).json["conversation_id"]
        self.assertEqual(self.client.delete(f"/api/conversations/{cid}").status_code, 204)
        self.assertEqual(self.client.get(f"/api/conversations/{cid}").status_code, 404)

    def test_login_required_and_logout(self):
        anon = chat_app.app.test_client()
        self.assertEqual(anon.get("/api/conversations").status_code, 401)
        self.assertEqual(anon.post("/chat", json={"text": "hi"}).status_code, 401)
        self.assertEqual(anon.get("/").headers["Location"], "/login")
        self.assertEqual(anon.get("/login").status_code, 200)

        self.assertEqual(self.client.get("/").status_code, 200)
        self.client.post("/logout")
        self.assertEqual(self.client.get("/api/conversations").status_code, 401)

    def test_login_and_register_validation(self):
        anon = chat_app.app.test_client()
        bad = anon.post("/login", data={"email": "pedro@example.com", "password": "wrong-pass"})
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(anon.post("/login", data={"email": "nobody@example.com", "password": "x"}).status_code, 401)
        self.assertEqual(anon.post("/register", data={"email": "PEDRO@example.com", "password": "password123"}).status_code, 400)
        self.assertEqual(anon.post("/register", data={"email": "not-an-email", "password": "password123"}).status_code, 400)
        self.assertEqual(anon.post("/register", data={"email": "ana@example.com", "password": "short"}).status_code, 400)
        self.assertEqual(anon.post("/login", data={"email": "pedro@example.com", "password": "password123"}).status_code, 302)
        self.assertEqual(anon.get("/api/conversations").status_code, 200)

    @patch.object(chat_app.bedrock, "converse", return_value=fake_reply("ok"))
    def test_free_limit(self, converse):
        for _ in range(chat_app.FREE_MESSAGE_LIMIT):
            self.assertEqual(self.client.post("/chat", json={"text": "hi"}).status_code, 200)
        # Disabled by default: message 11 still goes through (and is counted).
        self.assertEqual(self.client.post("/chat", json={"text": "hi"}).status_code, 200)

        with patch.object(chat_app, "ENFORCE_FREE_LIMIT", True):
            r = self.client.post("/chat", json={"text": "hi"})
            self.assertEqual(r.status_code, 402)
            calls = converse.call_count
            for c in self.client.get("/api/conversations").json:  # deleting chats must not reset the count
                self.client.delete(f"/api/conversations/{c['id']}")
            self.assertEqual(self.client.post("/chat", json={"text": "hi"}).status_code, 402)
            self.assertEqual(converse.call_count, calls)  # blocked before calling Bedrock

            db = chat_app.sqlite3.connect(self.db_path)
            db.execute("UPDATE users SET plan = 'paid'")
            db.commit()
            db.close()
            self.assertEqual(self.client.post("/chat", json={"text": "hi"}).status_code, 200)

    def test_email_validation(self):
        anon = chat_app.app.test_client()
        for bad in ["not-an-email", "a@b", "a b@c.com", "@c.com", "a@.com"]:
            r = anon.post("/register", data={"email": bad, "password": "password123"})
            self.assertEqual(r.status_code, 400, bad)
        r = anon.post("/register", data={"email": "07pm.dias@gmail.com", "password": "password123"})
        self.assertEqual(r.status_code, 302)

    @patch.object(chat_app.bedrock, "converse", return_value=fake_reply("ok"))
    def test_users_cannot_see_each_others_chats(self, _converse):
        cid = self.client.post("/chat", json={"text": "secret"}).json["conversation_id"]
        other = chat_app.app.test_client()
        other.post("/register", data={"email": "ana@example.com", "password": "password123"})
        self.assertEqual(other.get("/api/conversations").json, [])
        self.assertEqual(other.get(f"/api/conversations/{cid}").status_code, 404)
        self.assertEqual(other.post("/chat", json={"conversation_id": cid, "text": "hi"}).status_code, 404)
        self.assertEqual(other.delete(f"/api/conversations/{cid}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/conversations/{cid}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
