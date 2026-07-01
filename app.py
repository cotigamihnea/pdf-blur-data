import customtkinter as ctk
from tkinter import filedialog
import os
import threading
from redactor import DocumentRedactor

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")  

class PIIRedactorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Local Document Redactor (Romanian)")
        self.geometry("550x400") # Am lățit puțin fereastra pentru a încăpea textele mai lungi
        self.resizable(False, False)

        # Modificat: Acum păstrăm o listă de căi către fișiere, nu doar una
        self.file_paths = []
        self.status_label = None  
        self.redactor = None
        
        # --- UI Layout ---
        self.title_label = ctk.CTkLabel(self, text="PDF Data Blocker", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(30, 10))

        self.subtitle_label = ctk.CTkLabel(self, text="Selectează unul sau mai multe PDF-uri scanate.", text_color="gray")
        self.subtitle_label.pack(pady=(0, 20))

        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(pady=10, padx=20, fill="x")

        self.select_btn = ctk.CTkButton(self.file_frame, text="Răsfoiește PDF", command=self.select_files)
        self.select_btn.pack(side="left", padx=10, pady=10)

        self.file_label = ctk.CTkLabel(self.file_frame, text="Niciun fișier selectat", text_color="gray")
        self.file_label.pack(side="left", padx=10, pady=10)

        # --- Zona de acțiuni (Butoanele) ---
        self.actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.actions_frame.pack(pady=10)

        self.blur_data_btn = ctk.CTkButton(self.actions_frame, text="BLUREAZĂ DATE (PII)", command=self.start_data_thread, 
                                      fg_color="#b22222", hover_color="#8b0000", font=ctk.CTkFont(weight="bold"))
        self.blur_data_btn.pack(side="left", padx=10)

        self.blur_sig_btn = ctk.CTkButton(self.actions_frame, text="BLUREAZĂ SEMNĂTURI", command=self.start_signature_thread, 
                                      fg_color="#e59400", hover_color="#b37300", font=ctk.CTkFont(weight="bold"))
        self.blur_sig_btn.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(self, text="Se încarcă modelele AI... Te rugăm să aștepți.", text_color="cyan")
        self.status_label.pack(pady=15)
        
        threading.Thread(target=self.lazy_load_backend, daemon=True).start()

    def lazy_load_backend(self):
        try:
            self.redactor = DocumentRedactor()
            self.status_label.configure(text="Pregătit pentru procesare.", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Eroare la încărcare: {str(e)}", text_color="red")

    def select_files(self):
        filetypes = (("PDF files", "*.pdf"), ("All files", "*.*"))
        # askopenfilenames returnează o listă (tuplu) cu toate fișierele selectate
        paths = filedialog.askopenfilenames(title="Deschide PDF-uri", filetypes=filetypes)
        
        if paths:
            self.file_paths = list(paths)
            
            # Actualizăm textul de pe ecran în funcție de câte fișiere a ales
            if len(self.file_paths) == 1:
                display_text = os.path.basename(self.file_paths[0])
            else:
                display_text = f"{len(self.file_paths)} fișiere selectate"
                
            self.file_label.configure(text=display_text, text_color="white")
            
            if self.redactor:
                self.status_label.configure(text="Fișiere încărcate. Alege o acțiune.", text_color="green")

    def _prepare_processing(self):
        if not self.file_paths:
            self.status_label.configure(text="Eroare: Selectează cel puțin un PDF.", text_color="red")
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
        
        try:
            # Parcurgem fiecare fișier din lista selectată
            for index, current_path in enumerate(self.file_paths):
                filename = os.path.basename(current_path)
                
                # Actualizăm UI-ul să arate progresul (ex: 1/3)
                self.status_label.configure(
                    text=f"Procesare {index + 1}/{total_files}: {filename}...", 
                    text_color="orange"
                )
                self.update() # Forțăm interfața să se redeseneze ca să vezi textul nou
                
                # Trimitem fișierul curent la backend
                if mode == "data":
                    self.redactor.process_pdf(current_path)
                elif mode == "signatures":
                    self.redactor.process_signatures(current_path)
                    
            # Când bucla s-a terminat cu succes:
            self.status_label.configure(text=f"Succes! Am finalizat {total_files} document(e).", text_color="green")
            
        except Exception as e:
            self.status_label.configure(text=f"Eroare la fișierul {filename}: {str(e)}", text_color="red")
            
        finally:
            # Reactivăm butoanele la final, indiferent dacă a fost succes sau eroare
            self.blur_data_btn.configure(state="normal")
            self.blur_sig_btn.configure(state="normal")

if __name__ == "__main__":
    app = PIIRedactorApp()
    app.mainloop()