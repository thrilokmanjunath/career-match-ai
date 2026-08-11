import fitz  # PyMuPDF
import docx
import io

class ResumeParser:
    @staticmethod
    def parse_pdf(file_bytes: bytes) -> str:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc])
            return text.strip()
        except Exception as e:
            return ""

    @staticmethod
    def parse_docx(file_bytes: bytes) -> str:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            return ""

    @staticmethod
    def parse_txt(file_bytes: bytes) -> str:
        try:
            return file_bytes.decode("utf-8").strip()
        except Exception as e:
            return ""

    @classmethod
    def parse(cls, file_bytes: bytes, filename: str) -> str:
        ext = filename.lower().split('.')[-1]
        if ext == 'pdf':
            return cls.parse_pdf(file_bytes)
        elif ext in ['docx', 'doc']:
            return cls.parse_docx(file_bytes)
        elif ext == 'txt':
            return cls.parse_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
