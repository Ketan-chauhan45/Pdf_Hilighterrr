from flask import Flask, request, jsonify, render_template
from pdf_highlighter import Project
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        print(type(query))
        threshold = float(request.form.get("threshold", 0.2))

        # Process the uploaded files
        files = request.files.getlist("pdf_files")
        for file in files:
            if file.filename.endswith(".pdf"):
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))

        # Run the PDF processing project
        model = "sentence-transformers/all-MiniLM-L6-v2"
        project_instance = Project(app.config["UPLOAD_FOLDER"], model, query, threshold)
        result = project_instance.run()
        
        return jsonify({"relevant_pdfs": result})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)