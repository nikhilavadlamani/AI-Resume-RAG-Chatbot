# Deployment

Run the backend:

```bash
uvicorn app.main:app --reload
```

Run the frontend in a separate terminal:

```bash
streamlit run frontend/app.py
```

Optional environment variables:

- `HUGGINGFACEHUB_API_TOKEN`
- `API_BASE_URL`

If you containerize the app, start the FastAPI service before the Streamlit frontend.
