import customtkinter as ctk
from tkinter import filedialog
import os
import sys
import ctypes
import threading
from redactor import DocumentRedactor
from tkinterdnd2 import TkinterDnD, DND_FILES

# ==========================================
# FIX PENTRU TASKBAR-UL DIN WINDOWS
# ==========================================
# Forțăm sistemul de operare să trateze scriptul/executabilul ca pe o aplicație 
# de sine stătătoare, prevenind gruparea ei sub procesul generic Python și
# obligând Taskbar-ul să afișeze iconița noastră personalizată.
try:
    myappid = 'pdf_data_blocker.v1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

# ==========================================
# RUTINĂ DE MANAGEMENT AL RESURSELOR (PyInstaller)
# ==========================================
def resource_path(relative_path):
    """ Returnează calea absolută către resurse, funcționând nativ în Dev și în PyInstaller (_MEIPASS) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")  

# ==========================================
# CLASĂ HIBRIDĂ: CustomTkinter + Drag & Drop
# ==========================================
class TkinterDnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

# Moștenim din clasa noastră hibridă pentru a avea suport Drag & Drop nativ
class PIIRedactorApp(TkinterDnD_CTk):
    def __init__(self):
        super().__init__()

        # --- TITLUL ȘI GEOMETRIA FERESTREI ---
        self.title("PDF Data Blocker")
        self.geometry("550x460") 
        self.resizable(False, False)

        # --- APLICARE ICONIȚĂ CUSTOM IN TOP BAR ---
        icon_path = resource_path("icon.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception as e:
                print(f"Eroare la încărcarea iconiței grafice: {e}")
        # ------------------------------------------

        self.file_paths = []
        self.output_dir = None  
        self.status_label = None  
        self.redactor = None
        
        # --- ACTIVARE DRAG & DROP PE TOATĂ FEREASTRA ---
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop_files)
        
        # --- UI Layout ---
        self.title_label = ctk.CTkLabel(self, text="PDF Data Blocker", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(30, 10))

        self.subtitle_label = ctk.CTkLabel(self, text="Selectează sau trage (Drag & Drop) PDF-uri scanate aici.", text_color="gray")
        self.subtitle_label.pack(pady=(0, 15))

        # --- Zona 1: Selectare Fișiere ---
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(pady=5, padx=20, fill="x")

        self.select_btn = ctk.CTkButton(self.file_frame, text="Răsfoiește PDF", command=self.select_files)
        self.select_btn.pack(side="left", padx=10, pady=10)

        self.file_label = ctk.CTkLabel(self.file_frame, text="Niciun fișier selectat", text_color="gray")
        self.file_label.pack(side="left", padx=10, pady=10)

        # --- Zona 2: Selectare Folder Salvare ---
        self.dir_frame = ctk.CTkFrame(self)
        self.dir_frame.pack(pady=5, padx=20, fill="x")

        self.select_dir_btn = ctk.CTkButton(self.dir_frame, text="Alege Folder Salvare", command=self.select_output_dir, 
                                             fg_color="#4a4a4a", hover_color="#333333")
        self.select_dir_btn.pack(side="left", padx=10, pady=10)

        self.dir_label = ctk.CTkLabel(self.dir_frame, text="Salvare: Lângă original (implicit)", text_color="gray")
        self.dir_label.pack(side="left", padx=10, pady=10)

        # --- Zona 3: Acțiuni (Butoanele de blurare) ---
        self.actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.actions_frame.pack(pady=15)

        self.blur_data_btn = ctk.CTkButton(self.actions_frame, text="BLUREAZĂ DATE (PII)", command=self.start_data_thread, 
                                      fg_color="#b22222", hover_color="#8b0000", font=ctk.CTkFont(weight="bold"))
        self.blur_data_btn.pack(side="left", padx=10)

        self.blur_sig_btn = ctk.CTkButton(self.actions_frame, text="BLUREAZĂ SEMNĂTURI", command=self.start_signature_thread, 
                                      fg_color="#e59400", hover_color="#b37300", font=ctk.CTkFont(weight="bold"))
        self.blur_sig_btn.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(self, text="Se încarcă modelele AI... Te rugăm să aștepți.", text_color="cyan")
        self.status_label.pack(pady=10)
        
        threading.Thread(target=self.lazy_load_backend, daemon=True).start()

    # ==========================================
    # FUNCȚIA PENTRU DRAG & DROP
    # ==========================================
    def on_drop_files(self, event):
        dropped_files = self.tk.splitlist(event.data)
        pdf_files = [f for f in dropped_files if f.lower().endswith('.pdf')]
        
        if pdf_files:
            self.file_paths = pdf_files
            
            if len(self.file_paths) == 1:
                display_text = os.path.basename(self.file_paths[0])
            else:
                display_text = f"{len(self.file_paths)} fișiere selectate (Drag&Drop)"
                
            self.file_label.configure(text=display_text, text_color="white")
            
            if self.redactor:
                self.status_label.configure(text="Fișiere încărcate cu succes. Alege o acțiune.", text_color="green")
        else:
            self.status_label.configure(text="Eroare: Te rog să tragi doar fișiere PDF!", text_color="red")

    def lazy_load_backend(self):
        try:
            self.redactor = DocumentRedactor()
            self.status_label.configure(text="Pregătit pentru procesare.", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Eroare la încărcare: {str(e)}", text_color="red")

    def select_files(self):
        filetypes = (("PDF files", "*.pdf"), ("All files", "*.*"))
        paths = filedialog.askopenfilenames(title="Deschide PDF-uri", filetypes=filetypes)
        
        if paths:
            self.file_paths = list(paths)
            
            if len(self.file_paths) == 1:
                display_text = os.path.basename(self.file_paths[0])
            else:
                display_text = f"{len(self.file_paths)} fișiere selectate"
                
            self.file_label.configure(text=display_text, text_color="white")
            
            if self.redactor:
                self.status_label.configure(text="Fișiere încărcate. Alege o acțiune.", text_color="green")

    def select_output_dir(self):
        folder = filedialog.askdirectory(title="Alege unde se vor salva fișierele")
        if folder:
            self.output_dir = folder
            if len(folder) > 35:
                display_text = "..." + folder[-32:]
            else:
                display_text = folder
            self.dir_label.configure(text=f"Salvare în: {display_text}", text_color="white")

    def _prepare_processing(self):
        if not self.file_paths:
            self.status_label.configure(text="Eroare: Selectează sau trage cel puțin un PDF.", text_color="red")
            return False
        if not self.redactor:
            self.status_label.configure(text="Modelele se încarcă încă...", text_color="orange")
            return False
            
        self.blur_data_btn.configure(state="disabled")
        self.blur_sig_btn.configure(state="disabled")
        return True

    def start_data_thread(self):
        if self._prepare_processing():
            threading.Thread(target=self.process_documents, args=("data",), daemon=True).start()

    def start_signature_thread(self):
        if self._prepare_processing():
            threading.Thread(target=self.process_documents, args=("signatures",), daemon=True).start()

    def process_documents(self, mode):
        total_files = len(self.file_paths)
        filename = ""
        try:
            for index, current_path in enumerate(self.file_paths):
                filename = os.path.basename(current_path)
                
                self.status_label.configure(
                    text=f"Procesare {index + 1}/{total_files}: {filename}...", 
                    text_color="orange"
                )
                self.update() 
                
                if mode == "data":
                    self.redactor.process_pdf(current_path, output_dir=self.output_dir)
                elif mode == "signatures":
                    self.redactor.process_signatures(current_path, output_dir=self.output_dir)
                    
            self.status_label.configure(text=f"Succes! Am finalizat {total_files} document(e).", text_color="green")
            
        except Exception as e:
            self.status_label.configure(text=f"Eroare la fișierul {filename}: {str(e)}", text_color="red")
            
        finally:
            self.blur_data_btn.configure(state="normal")
            self.blur_sig_btn.configure(state="normal")

if __name__ == "__main__":
    app = PIIRedactorApp()
    app.mainloop()