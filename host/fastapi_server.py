from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from host.shared.parser_wrapper import parser
import shutil, os, json

app = FastAPI()

@app.post("/perform_de")
async def perform_de(pdf: UploadFile, config_name: str = Form(...)):
    dataset_path = os.path.join("../dataset", pdf.filename)
    output_path = os.path.join("../output", pdf.filename.replace(".pdf", ".json"))
    config_path = os.path.join("../de_config", config_name)

    os.makedirs("../dataset", exist_ok=True)
    os.makedirs("../output", exist_ok=True)

    if not os.path.exists(dataset_path):
        with open(dataset_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)

    try:
        # Load extraction configuration
        with open(config_path, 'r') as file:
            extraction_config = json.load(file)
        parser.perform_de(dataset_path, extraction_config, output_path)
        with open(output_path, 'r') as f:
            result = json.load(f)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)


