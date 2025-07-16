import random

from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(6, 10)

    @task
    def get_root(self) -> None:
        self.client.get("/")

    @task(3)
    def get_predict(self) -> None:
        sample_texts = [
            "The company's profits increased this quarter.",
            "Stock prices fell sharply after the announcement.",
            "The market outlook is positive.",
            "Investors are worried about inflation.",
            "Revenue growth exceeded expectations."
        ]

        data = {
            "text": random.choice(sample_texts)
        }

        self.client.post("/predict", json=data)
