from app.main import healthcheck


if __name__ == "__main__":
    print({"service": "resume-rag", "health": healthcheck().status})
