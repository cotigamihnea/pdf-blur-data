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
        """
        Sistem Hibrid 4D: YOLO False Positives + OCR + Culori CAD
        """
        # --- PAS 1: Inovația "YOLO False Positives" ---
        # Scanăm pagina cu modelul de ștampile, cu confidență foarte mică.
        # Pe o fațadă CAD, va prinde zeci de geamuri/balustrade. Pe un contract, va prinde maxim 2-3 lucruri.
        stamp_results = self.stamp_model(image, conf=0.15, imgsz=1536, verbose=False)
        raw_stamp_count = len(stamp_results[0].boxes)
        
        # --- PAS 2: Analiza OCR ---
        small_img_ocr = image.copy()
        small_img_ocr.thumbnail((2000, 2000))
        small_img_ocr = small_img_ocr.convert('L')
        text = pytesseract.image_to_string(small_img_ocr, lang='ron')
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', text)
        char_count = len(clean_text)
        
        # --- PAS 3: Analiza Cromatică Globală ---
        small_img_color = image.copy()
        small_img_color.thumbnail((1000, 1000)) 
        cv_img = cv2.cvtColor(np.array(small_img_color.convert('RGB')), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        
        mask_accepted_base = (s > 25) & (v > 60) & (
            ((h >= 11) & (h <= 99)) | 
            ((h >= 141) & (h <= 159)) 
        )

        mask_print_blue = (s > 60) & (v > 70) & (h >= 100) & (h <= 140)
        mask_print_red = (s > 60) & (v > 75) & (
            ((h >= 0) & (h <= 10)) | 
            ((h >= 160) & (h <= 180)) 
        )

        color_mask = mask_accepted_base | mask_print_blue | mask_print_red
        h_color = h[color_mask]
        
        cad_bins = [0, 0, 0, 0]
        if len(h_color) > 0:
            cad_bins[0] = np.sum((h_color >= 11) & (h_color <= 35))   
            cad_bins[1] = np.sum((h_color >= 36) & (h_color <= 85))   
            cad_bins[2] = np.sum((h_color >= 86) & (h_color <= 99))   
            cad_bins[3] = np.sum((h_color >= 141) & (h_color <= 159)) 
                
        cad_colors_found = sum(1 for count in cad_bins if count > 50)
                
        print(f"   [Analiză] OCR: {char_count} chars | Culori CAD: {cad_colors_found} | Ștampile Brute (YOLO): {raw_stamp_count}")
        
        # --- Decizia Supremă ---
        
        # 1. Plasa de Siguranță YOLO (Inovația Ta)
        if raw_stamp_count > 3:
            return True # O fațadă CAD sau un tabel masiv, sigur nu e text.
            
        # 2. Logică OCR + Culoare
        if char_count < 600:
            return True 
        elif char_count > 2000:
            return False 
        else:
            if cad_colors_found > 1:
                return True 
            else:
                return False 

    def _count_distinct_colors(self, crop_img):
        cv_img = cv2.cvtColor(np.array(crop_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]

        colorful_mask = (s_channel > 40) & (v_channel > 40)
        h_colorful = h_channel[colorful_mask]

        if len(h_colorful) < 50: 
            return 1 

        bins = [0, 0, 0, 0, 0] 
        
        for h in h_colorful:
            if (h <= 10) or (h >= 160): bins[0] += 1       
            elif 11 <= h <= 35: bins[1] += 1               
            elif 36 <= h <= 85: bins[2] += 1               
            elif 86 <= h <= 100: bins[3] += 1              
            elif 101 <= h <= 159: bins[4] += 1             

        total_colorful = len(h_colorful)
        distinct_colors = 0
        
        for count in bins:
            if count / total_colorful > 0.05: 
                distinct_colors += 1

        return distinct_colors

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
        cv_img = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        for (x1, y1, x2, y2) in excluded_boxes:
            cv2.rectangle(mask_blue, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
            
        blurred_mask = cv2.GaussianBlur(mask_blue, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred_mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2,          
            minDist=200,     
            param1=50,       
            param2=75,       
            minRadius=60,    
            maxRadius=300    
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"   [OpenCV] S-au detectat {len(circles)} cercuri albastre perfecte.")
            
            image_with_circles_pil = image.convert('RGB')
            
            for (center_x, center_y, radius) in circles:
                x1 = max(0, center_x - radius - 15)
                y1 = max(0, center_y - radius - 15)
                x2 = min(image.width, center_x + radius + 15)
                y2 = min(image.height, center_y + radius + 15)
                
                roi_mask = mask_blue[y1:y2, x1:x2]
                if roi_mask.size > 0:
                    color_ratio = cv2.countNonZero(roi_mask) / roi_mask.size
                    if color_ratio < 0.03:
                        continue 
                
                cropped_box = image.crop((x1, y1, x2, y2))
                w, h = cropped_box.size
                if w <= 0 or h <= 0: continue
                
                pixel_size = 18 
                small_w = max(1, w // pixel_size)
                small_h = max(1, h // pixel_size)
                if small_w < 1 or small_h < 1: continue
                
                small_img = cropped_box.resize((small_w, small_h), resample=Image.NEAREST)
                pixelated_img = small_img.resize((w, h), resample=Image.NEAREST)
                image_with_circles_pil.paste(pixelated_img, (x1, y1))
                
            return image_with_circles_pil
            
        return image 

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
            img_width, img_height = image.size
            
            # --- RUTAREA MODELELOR ---
            if is_schema:
                print(f"-> Pagina {i + 1}: SCHEMĂ (Analiză Cromatică + Margin Rule Dinamic)")
                stamp_results = self.stamp_model(image, conf=0.35, imgsz=1536, verbose=False)
                sig_results = self.sig_model(image, conf=0.05, imgsz=1536, verbose=False) 
                pipeline = [("stamp", stamp_results), ("sig", sig_results)]
            else:
                print(f"-> Pagina {i + 1}: PAGINĂ TEXT (Semnături Generale + Fallback Nume)")
                sig_results = self.sig_model(image, conf=0.05, imgsz=1536, verbose=False)
                pipeline = [("sig", sig_results)]
                
            max_w = img_width * 0.55  
            max_h = img_height * 0.55 
            yolo_boxes_applied = []
            
            for model_type, results in pipeline:
                for result in results:
                    for box, conf_tensor in zip(result.boxes.xyxy, result.boxes.conf):
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        conf = float(conf_tensor)
                        tight_w = x2 - x1
                        tight_h = y2 - y1
                        
                        if tight_w < max_w and tight_h < max_h:
                            
                            tight_crop_for_analysis = image.crop((x1, y1, x2, y2))
                            
                            if model_type == "stamp":
                                pad_x = 40
                                pad_y_top = 30
                                pad_y_bottom = 70 
                            else:
                                pad_x = 15
                                pad_y_top = 15
                                pad_y_bottom = 15
                                
                            crop_x1 = max(0, x1 - pad_x)
                            crop_y1 = max(0, y1 - pad_y_top)
                            crop_x2 = min(img_width, x2 + pad_x)
                            crop_y2 = min(img_height, y2 + pad_y_bottom)
                            
                            padded_w = crop_x2 - crop_x1
                            padded_h = crop_y2 - crop_y1
                            
                            if padded_w <= 0 or padded_h <= 0:
                                continue
                            
                            action = None 
                            
                            if is_schema:
                                if model_type == "stamp":
                                    valid_stamp = True
                                    aspect_ratio = tight_w / tight_h if tight_h > 0 else 1.0
                                    
                                    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                                        valid_stamp = False
                                        
                                    if valid_stamp:
                                        color_count = self._count_distinct_colors(tight_crop_for_analysis)
                                        if color_count >= 3: 
                                            print("   [INFO] Stampila refuzată: Prea multe culori (probabil desen CAD).")
                                            valid_stamp = False
                                            
                                    if valid_stamp:
                                        check_img = tight_crop_for_analysis.convert('L')
                                        check_img = ImageEnhance.Contrast(check_img).enhance(2.0)
                                        text_in_stamp = pytesseract.image_to_string(check_img, lang='ron').lower()
                                        
                                        forbidden_words = ["inventar", "coordonate", "stereografic", "pct", "parcela", "tabel"]
                                        if any(fw in text_in_stamp for fw in forbidden_words):
                                            print("   [INFO] Stampila refuzată: S-a detectat un tabel cadastral prin OCR.")
                                            valid_stamp = False
                                            
                                    if valid_stamp and conf < 0.80:
                                        if tight_crop_for_analysis.mode == 'RGB':
                                            full_np = np.array(tight_crop_for_analysis).astype(np.int16)
                                            r, g, b_chan = full_np[:,:,0], full_np[:,:,1], full_np[:,:,2]
                                            color_mask = (np.abs(r - g) > 25) | (np.abs(r - b_chan) > 25) | (np.abs(g - b_chan) > 25)
                                            if np.sum(color_mask) / color_mask.size < 0.005: 
                                                valid_stamp = False 
                                        else:
                                            valid_stamp = False
                                            
                                    if valid_stamp:
                                        action = "mosaic"

                                elif model_type == "sig":
                                    valid_sig = False
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    
                                    if img_width > img_height:
                                        margin_x_pct = 0.15  
                                        margin_y_pct = 0.20  
                                    else:
                                        margin_x_pct = 0.20  
                                        margin_y_pct = 0.12  
                                        
                                    margin_x = img_width * margin_x_pct
                                    margin_y = img_height * margin_y_pct
                                    
                                    if (center_x < margin_x or center_x > img_width - margin_x or 
                                        center_y < margin_y or center_y > img_height - margin_y):
                                        
                                        if tight_crop_for_analysis.mode == 'RGB':
                                            full_np = np.array(tight_crop_for_analysis).astype(np.int16)
                                            r, g, b_chan = full_np[:,:,0], full_np[:,:,1], full_np[:,:,2]
                                            is_blue = (b_chan > r + 10) & (b_chan > g + 10)
                                            is_red = (r > b_chan + 10) & (r > g + 10)
                                            color_mask = is_blue | is_red
                                            
                                            if np.sum(color_mask) / color_mask.size > 0.0005:
                                                valid_sig = True
                                            elif conf > 0.15:
                                                valid_sig = True
                                    else:
                                        valid_sig = False
                                        
                                    if valid_sig:
                                        action = "whiteout"
                            else:
                                action = "whiteout"
                                
                            if action == "mosaic":
                                cropped_box = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                                pixel_size = 18 
                                small_w = max(1, padded_w // pixel_size)
                                small_h = max(1, padded_h // pixel_size)
                                
                                if small_w >= 1 and small_h >= 1:
                                    small_img = cropped_box.resize((small_w, small_h), resample=Image.NEAREST)
                                    pixelated_img = small_img.resize((padded_w, padded_h), resample=Image.NEAREST)
                                    image.paste(pixelated_img, (crop_x1, crop_y1))
                                    yolo_boxes_applied.append((crop_x1, crop_y1, crop_x2, crop_y2))

                            elif action == "whiteout":
                                full_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
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
                                    if np.sum(color_mask) / color_mask.size > 0.0005: 
                                        is_colored = True
                                        
                                if is_colored:
                                    mask_img = Image.fromarray((color_mask * 255).astype(np.uint8), mode='L')
                                    expanded_mask = mask_img.filter(ImageFilter.MaxFilter(19))
                                    expanded_mask = expanded_mask.filter(ImageFilter.MaxFilter(11))
                                    expanded_np = np.array(expanded_mask)
                                    full_np[expanded_np == 255] = [255, 255, 255]
                                    cleaned_img = Image.fromarray(full_np.astype('uint8'))
                                    image.paste(cleaned_img, (crop_x1, crop_y1))
                                    yolo_boxes_applied.append((crop_x1, crop_y1, crop_x2, crop_y2))
                                    
                                else:
                                    padding = 12
                                    is_large_box = (tight_w > img_width * 0.18) or (tight_h > img_height * 0.10)
                                    if is_large_box:
                                        y1 = y1 + (tight_h * 0.45)
                                        
                                    crop_y1_safe = int(max(0, y1 - padding))
                                    crop_h_safe = crop_y2 - crop_y1_safe
                                    crop_w_safe = crop_x2 - crop_x1
                                    if crop_w_safe <= 0 or crop_h_safe <= 0: continue
                                    
                                    cropped_img = image.crop((crop_x1, crop_y1_safe, crop_x2, crop_y2))
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
                                        conf_text = int(ocr_data['conf'][j])
                                        clean_word = re.sub(r'[^a-zA-Z0-9ăâîșțĂÂÎȘȚ]', '', word)
                                        if len(clean_word) >= 2 and conf_text > 15:
                                            raw_boxes.append({
                                                'x': int(ocr_data['left'][j]), 'y': int(ocr_data['top'][j]),
                                                'w': int(ocr_data['width'][j]), 'h': int(ocr_data['height'][j])
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
                                                int(max(0, t_x - text_pad)), int(max(0, t_y - text_pad)),
                                                int(min(crop_w_safe, t_x + t_w + text_pad)), int(min(crop_h_safe, t_y + safe_h + text_pad))
                                            ))
                                    
                                    mask = Image.new("L", (crop_w_safe, crop_h_safe), 255)
                                    mask_draw = ImageDraw.Draw(mask)
                                    for (tx1, ty1, tx2, ty2) in valid_text_boxes:
                                        mask_draw.rectangle([tx1, ty1, tx2, ty2], fill=0)
                                    
                                    white_block = Image.new("RGB", (crop_w_safe, crop_h_safe), "white")
                                    image.paste(white_block, (crop_x1, crop_y1_safe), mask=mask)
                                    yolo_boxes_applied.append((crop_x1, crop_y1_safe, crop_x2, crop_y2))

            # ===============================================
            # PLASĂ DE SIGURANȚĂ PENTRU NUME PE PAGINI TEXT
            # ===============================================
            if not is_schema:
                print("   [Fallback Text] Se verifică după nume și ancore de semnătură ignorate de model...")
                ocr_fb = pytesseract.image_to_data(image, lang='ron', output_type=Output.DICT)
                fb_boxes = []
                for j in range(len(ocr_fb['text'])):
                    w_text = ocr_fb['text'][j].strip()
                    if w_text:
                        fb_boxes.append({
                            'text': w_text, 'left': int(ocr_fb['left'][j]), 'top': int(ocr_fb['top'][j]),
                            'width': int(ocr_fb['width'][j]), 'height': int(ocr_fb['height'][j])
                        })
                
                if fb_boxes:
                    fb_full_text = " ".join([b['text'] for b in fb_boxes])
                    
                    sig_keywords = {"secretar", "primar", "director", "șef", "sef", "întocmit", "intocmit", 
                                    "semnătura", "semnatura", "arhitect", "inginer", "expert", "verificator"}
                    
                    target_indices = set()
                    
                    fb_results = self.analyzer.analyze(text=fb_full_text, entities=["PERSON"], language='ro')
                    for res in fb_results:
                        c_idx = 0
                        for i, fb_box in enumerate(fb_boxes):
                            w_len = len(fb_box['text'])
                            if (c_idx < res.end) and (c_idx + w_len > res.start):
                                target_indices.add(i)
                            c_idx += w_len + 1
                            
                    for i, fb_box in enumerate(fb_boxes):
                        clean_w = fb_box['text'].lower().strip(" .,:;()[]{}\n\t'\"")
                        if clean_w in sig_keywords:
                            target_indices.add(i)
                            if i + 1 < len(fb_boxes): target_indices.add(i + 1)
                            if i + 2 < len(fb_boxes): target_indices.add(i + 2)
                    
                    for i in target_indices:
                        fb_box = fb_boxes[i]
                        clean_w = fb_box['text'].lower().strip(" .,:;()[]{}\n\t'\"")
                        if clean_w in self.ignore_list and clean_w not in sig_keywords:
                            continue
                            
                        fx1 = fb_box['left']
                        fy1 = fb_box['top']
                        fx2 = fx1 + fb_box['width']
                        fy2 = fy1 + fb_box['height']
                        
                        fcx = (fx1 + fx2) / 2
                        fcy = (fy1 + fy2) / 2
                        
                        in_yolo = False
                        for (yx1, yy1, yx2, yy2) in yolo_boxes_applied:
                            if yx1 <= fcx <= yx2 and yy1 <= fcy <= yy2:
                                in_yolo = True
                                break
                        
                        if not in_yolo:
                            pad_x = 40
                            pad_y_top = 20
                            pad_y_bottom = 80 
                            
                            ax1 = max(0, fx1 - pad_x)
                            ay1 = max(0, fy1 - pad_y_top)
                            ax2 = min(img_width, fx2 + pad_x)
                            ay2 = min(img_height, fy2 + pad_y_bottom)
                            
                            c_crop = image.crop((ax1, ay1, ax2, ay2))
                            if c_crop.mode == 'RGB':
                                f_np = np.array(c_crop).astype(np.int16)
                                cr = f_np[:,:,0]
                                cg = f_np[:,:,1]
                                cb = f_np[:,:,2]
                                
                                c_mask = (np.abs(cr - cg) > 15) | (np.abs(cr - cb) > 15) | (np.abs(cg - cb) > 15)
                                
                                if np.sum(c_mask) > 10: 
                                    box_h = fy2 - fy1
                                    
                                    bx1 = max(0, fx1 - 15)
                                    by1 = max(0, fy1 - 10)
                                    bx2 = min(img_width, fx2 + 15)
                                    by2 = min(img_height, fy2 + max(50, box_h * 3)) 
                                    
                                    cw = bx2 - bx1
                                    ch = by2 - by1
                                    if cw > 0 and ch > 0:
                                        white_block = Image.new("RGB", (cw, ch), "white")
                                        image.paste(white_block, (bx1, by1))
                                        yolo_boxes_applied.append((bx1, by1, bx2, by2))

            # Pasul 2: Procesare secundară OpenCV 
            if is_schema:
                image = self._process_schematic_circles(image, yolo_boxes_applied)
            
            redacted_images.append(image)
            
        print("Se salvează PDF-ul fără semnături/ștampile...")
        if redacted_images:
            redacted_images[0].save(output_pdf_path, save_all=True, append_images=redacted_images[1:])
        return output_pdf_path