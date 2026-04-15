from app.main import healthcheck


if __name__ == "__main__":
    print(healthcheck().model_dump())
