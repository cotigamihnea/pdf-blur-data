import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from ultralytics import YOLO
import numpy as np
import cv2
import os
import sys
import re

class DocumentRedactor:
    def __init__(self):
        print("Inițializare sisteme portabile...")
        
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        tesseract_exe = os.path.join(self.base_path, "tesseract", "tesseract.exe")
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        
        self.poppler_bin_path = os.path.join(self.base_path, "poppler", "Library", "bin")

        # ==========================================
        # 1. INITIALIZARE AI DATE PERSONALE
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
        # 2. INITIALIZARE AI VIZUAL (YOLO x2)
        # ==========================================
        print("-> Se încarcă modelele vizuale (YOLOv8)...")
        model_sig_path = os.path.join(self.base_path, "signature_detector.pt")
        model_stamp_path = os.path.join(self.base_path, "stamp_detector.pt")
        
        if not os.path.exists(model_sig_path):
            raise FileNotFoundError(f"Eroare critică: Modelul de semnături '{model_sig_path}' nu a fost găsit!")
        if not os.path.exists(model_stamp_path):
            raise FileNotFoundError(f"Eroare critică: Modelul de ștampile '{model_stamp_path}' nu a fost găsit!")
            
        self.sig_model = YOLO(model_sig_path)
        self.stamp_model = YOLO(model_stamp_path)

    def _is_schematic_page(self, image):
        small_img = image.copy()
        small_img.thumbnail((1000, 1000))
        small_img = small_img.convert('L')
        text = pytesseract.image_to_string(small_img, lang='ron')
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', text)
        char_count = len(clean_text)
        print(f"   [Scanare Rapidă OCR] S-au detectat {char_count} caractere valide.")
        return char_count < 400

    def process_pdf(self, input_pdf_path, output_dir=None):
        nume_fisier = os.path.basename(input_pdf_path)
        nume_nou = nume_fisier.replace(".pdf", "_DATE_BLURATE.pdf")
        if output_dir:
            output_pdf_path = os.path.join(output_dir, nume_nou)
        else:
            output_pdf_path = input_pdf_path.replace(".pdf", "_DATE_BLURATE.pdf")
        
        print("Se convertește PDF-ul în imagini...")
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
            if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-].*-[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', word) or \
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

    def _process_schematic_circles(self, image, excluded_boxes):
        """
        OPENCV ULTRA-STRICT PENTRU CERCURI ALBASTRE
        Acționează ca un lunetist doar după ce YOLO și-a terminat treaba.
        """
        cv_img = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # 1. Mască EXCLUSIV pentru ALBASTRU (ștampila ratată e clar albastră)
        # Tot ce e negru (tabelul) va dispărea!
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 2. Ștergem din viziunea lui OpenCV zonele deja cenzurate de YOLO!
        for (x1, y1, x2, y2) in excluded_boxes:
            # Colorăm cu negru (0) în mască cutiile pe care YOLO le-a rezolvat deja
            cv2.rectangle(mask_blue, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
            
        # Un mic blur pentru a face cercul mai solid
        blurred_mask = cv2.GaussianBlur(mask_blue, (9, 9), 2)
        
        # 3. HOUGH CIRCLES: Parametrii ridicați pentru PERFECȚIUNE
        circles = cv2.HoughCircles(
            blurred_mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2,          
            minDist=200,     
            param1=50,       
            param2=75,       # CRITIC: Crescut la 75! Acceptă DOAR cercuri aproape perfecte, respinge liniile de cadastru
            minRadius=60,    
            maxRadius=300    
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"   [OpenCV] S-au detectat {len(circles)} cercuri albastre perfecte (după eliminarea YOLO).")
            
            image_with_circles_pil = image.convert('RGB')
            
            for (center_x, center_y, radius) in circles:
                x1 = max(0, center_x - radius - 15)
                y1 = max(0, center_y - radius - 15)
                x2 = min(image.width, center_x + radius + 15)
                y2 = min(image.height, center_y + radius + 15)
                
                # 4. Verificare suplimentară: Măsurăm cât tuș albastru e efectiv acolo
                roi_mask = mask_blue[y1:y2, x1:x2]
                if roi_mask.size > 0:
                    color_ratio = cv2.countNonZero(roi_mask) / roi_mask.size
                    # Dacă are sub 3% albastru, e doar o iluzie optică formată din linii
                    if color_ratio < 0.03:
                        continue 
                
                cropped_box = image.crop((x1, y1, x2, y2))
                w, h = cropped_box.size
                
                if w <= 0 or h <= 0:
                    continue
                
                pixel_size = 18 
                small_w = max(1, w // pixel_size)
                small_h = max(1, h // pixel_size)
                
                if small_w < 1 or small_h < 1:
                    continue
                
                small_img = cropped_box.resize((small_w, small_h), resample=Image.NEAREST)
                pixelated_img = small_img.resize((w, h), resample=Image.NEAREST)
                
                image_with_circles_pil.paste(pixelated_img, (x1, y1))
                
            return image_with_circles_pil
            
        return image 

    def process_signatures(self, input_pdf_path, output_dir=None):
        nume_fisier = os.path.basename(input_pdf_path)
        nume_nou = nume_fisier.replace(".pdf", "_FARA_SEMNATURI.pdf")
        
        if output_dir:
            output_pdf_path = os.path.join(output_dir, nume_nou)
        else:
            output_pdf_path = input_pdf_path.replace(".pdf", "_FARA_SEMNATURI.pdf")
        
        print(f"\nSe procesează: {nume_fisier}")
        images = convert_from_path(input_pdf_path, poppler_path=self.poppler_bin_path)
        
        redacted_images = []
        for i, image in enumerate(images):
            
            is_schema = self._is_schematic_page(image)
            
            if is_schema:
                print(f"-> Pagina {i + 1}: SCHEMĂ (YOLO conf=0.40 prioritar)")
                results = self.stamp_model(image, conf=0.40, imgsz=1536, verbose=False)
            else:
                print(f"-> Pagina {i + 1}: PAGINĂ TEXT (YOLO conf=0.05)")
                results = self.sig_model(image, conf=0.05, imgsz=1536, verbose=False)
                
            img_width, img_height = image.size
            max_w = img_width * 0.55  
            max_h = img_height * 0.55 
            
            # Aici salvăm cutiile găsite de YOLO pentru a le pasa lui OpenCV
            yolo_boxes_applied = []
            
            # Pasul 1: Procesare PRIORITARĂ prin YOLO
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    if box_w < max_w and box_h < max_h:
                        padding = 12 
                        
                        crop_x1 = int(max(0, x1 - padding))
                        crop_y1_original = int(max(0, y1 - padding))
                        crop_x2 = int(min(img_width, x2 + padding))
                        crop_y2 = int(min(img_height, y2 + padding))
                        
                        w = crop_x2 - crop_x1
                        h = crop_y2 - crop_y1_original
                        
                        if w <= 0 or h <= 0:
                            continue
                        
                        if is_schema:
                            # Salvăm coordonatele dilatate ca OpenCV să le ignore mai târziu
                            yolo_boxes_applied.append((crop_x1, crop_y1_original, crop_x2, crop_y2))
                            
                            cropped_box = image.crop((crop_x1, crop_y1_original, crop_x2, crop_y2))
                            pixel_size = 18 
                            small_w = max(1, w // pixel_size)
                            small_h = max(1, h // pixel_size)
                            
                            if small_w < 1 or small_h < 1:
                                continue
                                
                            small_img = cropped_box.resize((small_w, small_h), resample=Image.NEAREST)
                            pixelated_img = small_img.resize((w, h), resample=Image.NEAREST)
                            image.paste(pixelated_img, (crop_x1, crop_y1_original))
                            
                        else:
                            full_crop = image.crop((crop_x1, crop_y1_original, crop_x2, crop_y2))
                            full_np = np.array(full_crop)
                            
                            is_colored = False
                            if full_crop.mode == 'RGB':
                                crop_int = full_np.astype(np.int16)
                                r = crop_int[:,:,0]
                                g = crop_int[:,:,1]
                                b = crop_int[:,:,2]
                                
                                is_blue = (b > r + 10) & (b > g + 10)
                                is_red = (r > b + 10) & (r > g + 10)
                                color_mask = is_blue | is_red
                                
                                if np.sum(color_mask) / color_mask.size > 0.002: 
                                    is_colored = True
                                    
                            if is_colored:
                                mask_img = Image.fromarray((color_mask * 255).astype(np.uint8), mode='L')
                                expanded_mask = mask_img.filter(ImageFilter.MaxFilter(19))
                                expanded_mask = expanded_mask.filter(ImageFilter.MaxFilter(11))
                                expanded_np = np.array(expanded_mask)
                                full_np[expanded_np == 255] = [255, 255, 255]
                                cleaned_img = Image.fromarray(full_np.astype('uint8'))
                                image.paste(cleaned_img, (crop_x1, crop_y1_original))
                                
                            else:
                                is_large_box = (box_w > img_width * 0.18) or (box_h > img_height * 0.10)
                                if is_large_box:
                                    y1 = y1 + (box_h * 0.45)
                                    box_h = y2 - y1 
                                    
                                crop_y1 = int(max(0, y1 - padding))
                                crop_h = crop_y2 - crop_y1
                                crop_w = crop_x2 - crop_x1
                                
                                if crop_w <= 0 or crop_h <= 0:
                                    continue
                                
                                cropped_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                                
                                if cropped_img.mode == 'RGB':
                                    r, g, b_channel = cropped_img.split()
                                    ocr_img = b_channel
                                else:
                                    ocr_img = cropped_img.convert('L')
                                    
                                enhancer = ImageEnhance.Contrast(ocr_img)
                                ocr_img = enhancer.enhance(2.5)
                                
                                ocr_data = pytesseract.image_to_data(ocr_img, lang='ron', output_type=Output.DICT)
                                
                                raw_boxes = []
                                for j, word in enumerate(ocr_data['text']):
                                    word = word.strip()
                                    conf = int(ocr_data['conf'][j])
                                    clean_word = re.sub(r'[^a-zA-Z0-9ăâîșțĂÂÎȘȚ]', '', word)
                                    
                                    if len(clean_word) >= 2 and conf > 15:
                                        raw_boxes.append({
                                            'x': int(ocr_data['left'][j]),
                                            'y': int(ocr_data['top'][j]),
                                            'w': int(ocr_data['width'][j]),
                                            'h': int(ocr_data['height'][j])
                                        })
                                
                                valid_text_boxes = []
                                if raw_boxes:
                                    heights = sorted([b['h'] for b in raw_boxes])
                                    median_h = heights[len(heights) // 2]
                                    
                                    for b in raw_boxes:
                                        t_x, t_y, t_w, t_h = b['x'], b['y'], b['w'], b['h']
                                        
                                        max_allowed_h = int(median_h * 1.5)
                                        safe_h = min(t_h, max_allowed_h)
                                        
                                        text_pad = 3
                                        valid_text_boxes.append((
                                            int(max(0, t_x - text_pad)),
                                            int(max(0, t_y - text_pad)),
                                            int(min(crop_w, t_x + t_w + text_pad)),
                                            int(min(crop_h, t_y + safe_h + text_pad))
                                        ))
                                
                                mask = Image.new("L", (crop_w, crop_h), 255)
                                mask_draw = ImageDraw.Draw(mask)
                                
                                for (tx1, ty1, tx2, ty2) in valid_text_boxes:
                                    mask_draw.rectangle([tx1, ty1, tx2, ty2], fill=0)
                                
                                white_block = Image.new("RGB", (crop_w, crop_h), "white")
                                image.paste(white_block, (crop_x1, crop_y1), mask=mask)

            # ========================================================
            # Pasul 2: Procesare secundară OpenCV doar cu ce a rămas (SCHEME)
            # ========================================================
            if is_schema:
                # Transmitem imaginea cenzurată și cutiile lui YOLO
                image = self._process_schematic_circles(image, yolo_boxes_applied)
            
            redacted_images.append(image)
            
        print("Se salvează PDF-ul fără semnături/ștampile...")
        if redacted_images:
            redacted_images[0].save(output_pdf_path, save_all=True, append_images=redacted_images[1:])
        return output_pdf_path