from sentence_transformers import SentenceTransformer
import torch
import os
from PyPDF2 import PdfReader

class Project:
    def __init__(self, pdf_dir, model_path, query, threshold):
        self.pdf_dir = pdf_dir
        self.model_path = model_path
        self.query = query
        self.threshold = threshold
        self.pdf_texts = []
        self.pdf_files = []

    def pdfs_to_list(self):
        self.pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        for pdf_file in self.pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            pdf_reader = PdfReader(pdf_path)
            text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            self.pdf_texts.append(text)
            print(f'Processing:{pdf_file}')
        return self.pdf_texts

    def make_embeddings(self, texts):
        model = SentenceTransformer(self.model_path)
        pdf_emb = model.encode(texts, convert_to_tensor=True)
        q_emb = model.encode(self.query, convert_to_tensor=True)
        return pdf_emb, q_emb

    def nearest_pdfs(self, pdf_emb, q_emb):
        similarity = torch.nn.functional.cosine_similarity(pdf_emb, q_emb.unsqueeze(0), dim=1)
        similarity_list = similarity.tolist()
        matching_indices = [i for i, sim in enumerate(similarity_list) if sim >= self.threshold]
        print(similarity)
        return [self.pdf_files[i] for i in matching_indices]

    def run(self):
        pdf_texts = self.pdfs_to_list()
        pdf_emb, q_emb = self.make_embeddings(pdf_texts)
        nearest_files = self.nearest_pdfs(pdf_emb, q_emb)
        return nearest_files

# Example usage
# input = "System"
# pdf_path = r"C:\Users\mr kaif\Downloads\sample_pdf"
# model = "sentence-transformers/all-MiniLM-L6-v2"
# project_instance = Project(pdf_path, model, input, threshold=0.2)
# result = project_instance.run()
# print(result)