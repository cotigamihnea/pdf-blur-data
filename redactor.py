import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from PIL import ImageDraw
from ultralytics import YOLO
import os
import sys
import re

class DocumentRedactor:
    def __init__(self):
        print("Inițializare sisteme portabile...")
        
        # --- DETERMINARE CALE EXECUTABIL (PyInstaller) ---
        # Dacă rulează ca .exe, sys._MEIPASS indică folderul temporar unde PyInstaller extrage fișierele
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # --- CONFIGURARE TRSEE PORTABILE WINDOWS ---
        tesseract_exe = os.path.join(self.base_path, "tesseract", "tesseract.exe")
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        
        # Calea către binarele Poppler (folosită la convert_from_path)
        self.poppler_bin_path = os.path.join(self.base_path, "poppler", "Library", "bin")

        # ==========================================
        # 1. INITIALIZARE AI DATE PERSONALE (NLP)
        # ==========================================
        print("-> Se încarcă motorul de text (Presidio/spaCy)...")
        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "ro", "model_name": "ro_core_news_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=config)
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["ro"])
        
        cnp_pattern = Pattern(name="cnp_regex", regex=r"\b[1-9]\d{12}\b", score=0.9)
        self.analyzer.registry.add_recognizer(PatternRecognizer(supported_entity="RO_CNP", patterns=[cnp_pattern]))

        self.ignore_list = {
            "tutore", "tutorele", "supervizor", "supervizorul", "prenume", "nume", 
            "practicant", "practicantul", "partener", "practica", "practică", 
            "semnatura", "semnătura", "director", "executiv", "student", "cadrul", 
            "didactic", "funcţia", "funcția", "drepturi", "şi", "de", "la", "din", "pentru"
        }
        
        self.anchors_2_words = {"nume", "numele", "prenume", "prenumele", "subsemnatul", "subsemnata"}
        self.anchors_1_word = {"cnp", "telefon", "tel", "email", "mail", "serie", "seria", "nr", "numar", "numărul"}

        # ==========================================
        # 2. INITIALIZARE AI SEMNĂTURI (YOLOv8)
        # ==========================================
        print("-> Se încarcă modelul vizual pentru semnături (YOLOv8)...")
        model_absolute_path = os.path.join(self.base_path, "signature_detector.pt")
        
        if not os.path.exists(model_absolute_path):
            raise FileNotFoundError(f"Eroare critică: Modelul '{model_absolute_path}' nu a fost găsit!")
            
        self.sig_model = YOLO(model_absolute_path)

    def process_pdf(self, input_pdf_path):
        output_pdf_path = input_pdf_path.replace(".pdf", "_DATE_BLURATE.pdf")
        
        print("Se convertește PDF-ul în imagini...")
        # Adăugat poppler_path din folderul nostru portabil
        images = convert_from_path(input_pdf_path, poppler_path=self.poppler_bin_path)
        
        redacted_images = []
        for i, image in enumerate(images):
            print(f"Se procesează textul pe pagina {i + 1}...")
            redacted_images.append(self._redact_image_data(image))
            
        print("Se salvează PDF-ul...")
        if redacted_images:
            redacted_images[0].save(output_pdf_path, save_all=True, append_images=redacted_images[1:])
        return output_pdf_path

    def _redact_image_data(self, image):
        ocr_data = pytesseract.image_to_data(image, lang='ron', output_type=Output.DICT)
        draw = ImageDraw.Draw(image)
        
        boxes = []
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            if word:
                boxes.append({
                    'text': word, 'left': ocr_data['left'][i], 'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i], 'height': ocr_data['height'][i]
                })
                
        full_text = " ".join([b['text'] for b in boxes])
        indices_to_blur = set()
        
        results = self.analyzer.analyze(text=full_text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "RO_CNP"], language='ro')
        for result in results:
            current_char_idx = 0
            for i, box in enumerate(boxes):
                word_len = len(box['text'])
                if (current_char_idx < result.end) and (current_char_idx + word_len > result.start):
                    clean_word = box['text'].lower().strip(" .,:;()[]{}\n\t'\"")
                    if clean_word not in self.ignore_list:
                        if not (re.match(r'^[\d\.\-\/]+$', clean_word) and result.entity_type not in ["PHONE_NUMBER", "RO_CNP"]):
                            indices_to_blur.add(i)
                current_char_idx += word_len + 1

        words_to_blur_count = 0
        for i, box in enumerate(boxes):
            word = box['text']
            clean_word = word.lower().strip(" .,:;()[]{}_-")
            
            if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', word) or \
               re.search(r'\b[1-9]\d{12}\b', word) or \
               re.search(r'\b(?:07|02|03)\d{8}\b', clean_word):
                indices_to_blur.add(i)
                continue
                
            if words_to_blur_count > 0:
                if clean_word not in self.anchors_1_word and clean_word not in self.anchors_2_words and clean_word not in self.ignore_list:
                    indices_to_blur.add(i)
                    words_to_blur_count -= 1
                else:
                    words_to_blur_count = 0 
            
            if i not in indices_to_blur:
                if clean_word in self.anchors_2_words:
                    words_to_blur_count = 2
                elif clean_word in self.anchors_1_word:
                    words_to_blur_count = 1

        for i in indices_to_blur:
            box = boxes[i]
            pad = 2 
            draw.rectangle(
                [box['left']-pad, box['top']-pad, box['left']+box['width']+pad, box['top']+box['height']+pad],
                fill="black"
            )
            
        return image

    def process_signatures(self, input_pdf_path):
        output_pdf_path = input_pdf_path.replace(".pdf", "_FARA_SEMNATURI.pdf")
        
        print("Se convertește PDF-ul în imagini...")
        # Adăugat poppler_path din folderul nostru portabil
        images = convert_from_path(input_pdf_path, poppler_path=self.poppler_bin_path)
        
        redacted_images = []
        for i, image in enumerate(images):
            print(f"Caut semnături vizuale pe pagina {i + 1}...")
            
            results = self.sig_model(image, conf=0.08, imgsz=1024, verbose=False)
            draw = ImageDraw.Draw(image)
            
            img_width, img_height = image.size
            max_w = img_width * 0.40  
            max_h = img_height * 0.25 
            
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    if box_w < max_w and box_h < max_h:
                        padding = 10 
                        draw.rectangle(
                            [x1 - padding, y1 - padding, x2 + padding, y2 + padding], 
                            fill="black"
                        )
            
            redacted_images.append(image)
            
        print("Se salvează PDF-ul fără semnături...")
        if redacted_images:
            redacted_images[0].save(output_pdf_path, save_all=True, append_images=redacted_images[1:])
        return output_pdf_path