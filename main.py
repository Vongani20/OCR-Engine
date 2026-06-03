# main.py - Fixed with correct model names and security
import os
import json
import hashlib
import logging
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# New SDK import
from google import genai
from google.genai import types
from PIL import Image

# Import PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install PyMuPDF")
    exit(1)

from pydantic import BaseModel, Field, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential

# ========== Configuration ==========
class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "input" / "invoices"
    PROCESSED_DIR = BASE_DIR / "input" / "processed"
    OUTPUT_DATA_DIR = BASE_DIR / "output" / "extracted_data"
    OUTPUT_REPORT_DIR = BASE_DIR / "output" / "combined_report"
    LOG_DIR = BASE_DIR / "logs"
    
    # Gemini settings 
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyA7f18YPugfof9ILiMqRkmYApqMIqaYh4k")
    
    # CORRECT MODEL NAMES for the API
    MODEL_NAMES = [
        "gemini-3-flash",       
        "gemini-2.0-flash",      
        "gemini-1.5-flash",      
        "gemini-3.5-flash",        
    ]
    MODEL_NAME = MODEL_NAMES[0]  # Start with gemini-3-flash
    
    # Processing settings
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf'}
    PDF_DPI = 200
    MAX_WORKERS = 1

# Create directories
for dir_path in [Config.INPUT_DIR, Config.PROCESSED_DIR, Config.OUTPUT_DATA_DIR, 
                 Config.OUTPUT_REPORT_DIR, Config.LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_DIR / f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== Pydantic Models (Your Schema) ==========
class Vendor(BaseModel):
    model_config = ConfigDict(extra="allow")
    company_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class Customer(BaseModel):
    model_config = ConfigDict(extra="allow")
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None

class LineItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    item_id: Optional[str] = None
    description: str
    quantity: float
    unit_price: float
    discount: float = 0.0
    taxable: bool = True
    total_price: float

class Totals(BaseModel):
    model_config = ConfigDict(extra="allow")
    subtotal: float
    tax_rate_percentage: float = 0.0
    tax_amount: float = 0.0
    discount_total: float = 0.0
    grand_total: float
    amount_paid: float = 0.0
    balance_due: float = 0.0

class PaymentDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: Optional[str] = None
    transaction_reference: Optional[str] = None
    terms: Optional[str] = None

class Invoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    invoice_number: Optional[str] = None
    status: Optional[str] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    payment_date: Optional[str] = None
    currency: str = "USD"
    vendor: Vendor
    customer: Customer
    line_items: List[LineItem]
    totals: Totals
    payment_details: Optional[PaymentDetails] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, alias='_metadata')

# ========== Custom JSON Encoder ==========
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# ========== Gemini Initialization ==========
if not Config.GEMINI_API_KEY:
    logger.error("ERROR: GEMINI_API_KEY not set!")
    logger.error("  Set it using: set GEMINI_API_KEY=your_key_here (Windows)")
    logger.error("  or: export GEMINI_API_KEY='your_key_here' (Linux/Mac)")
    logger.error("  NEVER hardcode API keys in your script!")
    exit(1)

client = genai.Client(api_key=Config.GEMINI_API_KEY)

def test_model_availability(model_name: str) -> bool:
    """Test if a model is available"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="OK",
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=5
            )
        )
        logger.info(f"✓ Model {model_name} is available")
        return True
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            logger.warning(f"Model {model_name} not found")
        else:
            logger.warning(f"Model {model_name} error: {e}")
        return False

# Find working model
logger.info("=" * 50)
logger.info("Testing available models...")
logger.info("=" * 50)

working_model = None
for model_name in Config.MODEL_NAMES:
    logger.info(f"Testing: {model_name}")
    if test_model_availability(model_name):
        working_model = model_name
        logger.info(f"✓ USING MODEL: {working_model}")
        break

if not working_model:
    logger.error("=" * 50)
    logger.error("No working model found!")
    logger.error("")
    logger.error("Please check:")
    logger.error("1. Your API key is valid")
    logger.error("2. You have internet connection")
    logger.error("3. The API key has Gemini API enabled")
    logger.error("")
    logger.error("Run this to list available models:")
    logger.error("python -c \"from google import genai; c=genai.Client(); [print(m.name) for m in c.models.list()]\"")
    logger.error("=" * 50)
    exit(1)

Config.MODEL_NAME = working_model

# ========== Invoice Extractor Class ==========
class InvoiceExtractor:
    def __init__(self):
        self.processed_files = set()
        self.invoices_data = []
        self.processing_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now()
        }
    
    def load_processed_files(self) -> set:
        processed_file = Config.OUTPUT_REPORT_DIR / "processed_files.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()
    
    def save_processed_files(self):
        processed_file = Config.OUTPUT_REPORT_DIR / "processed_files.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.processed_files), f, indent=2)
    
    def get_file_hash(self, file_path: Path) -> str:
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return str(datetime.now().timestamp())
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF pages to PIL Images using PyMuPDF"""
        images = []
        try:
            pdf_document = fitz.open(pdf_path)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            
            pdf_document.close()
            logger.info(f"  Converted {len(images)} pages")
            
        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            raise
        
        return images
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract_invoice_from_image(self, image: Image.Image, filename: str) -> Dict:
        """Extract invoice data using Gemini"""
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_bytes = img_byte_arr.getvalue()
        
        prompt = """Extract invoice data from this image. Return ONLY valid JSON.

Schema:
{
  "invoice": {
    "invoice_number": "string or null",
    "status": "Paid/Pending/Overdue or null",
    "issue_date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "payment_date": "YYYY-MM-DD or null",
    "currency": "USD/EUR/GBP/etc",
    "vendor": {
      "company_name": "string or null",
      "address": "string or null",
      "city": "string or null",
      "state": "string or null",
      "postal_code": "string or null",
      "country": "string or null",
      "email": "string or null",
      "phone": "string or null"
    },
    "customer": {
      "company_name": "string or null",
      "contact_name": "string or null",
      "address": "string or null",
      "city": "string or null",
      "state": "string or null",
      "postal_code": "string or null",
      "country": "string or null",
      "email": "string or null"
    },
    "line_items": [
      {
        "item_id": "string or null",
        "description": "string",
        "quantity": number,
        "unit_price": number,
        "discount": number,
        "taxable": boolean,
        "total_price": number
      }
    ],
    "totals": {
      "subtotal": number,
      "tax_rate_percentage": number,
      "tax_amount": number,
      "discount_total": number,
      "grand_total": number,
      "amount_paid": number,
      "balance_due": number
    },
    "payment_details": {
      "method": "string or null",
      "transaction_reference": "string or null",
      "terms": "string or null"
    },
    "notes": "string or null"
  }
}

Rules:
- Use null for missing fields
- Dates in YYYY-MM-DD format
- All monetary values as numbers
- Calculate missing totals
- Status: "Paid" if payment_date exists, "Pending" if due_date > today, else "Overdue"
"""
        
        # Gemini API call
        response = client.models.generate_content(
            model=Config.MODEL_NAME,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse response
        try:
            clean_response = response.text.strip()
            # Remove markdown code blocks
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.startswith('```'):
                clean_response = clean_response[3:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            
            data = json.loads(clean_response.strip())
            
            if "invoice" not in data:
                data = {"invoice": data}
            
            # Validate
            Invoice(**data["invoice"])
            return data
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            raise
    
    def process_pdf(self, pdf_path: Path, filename: str) -> List[Dict]:
        """Process PDF file"""
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            pages = self.pdf_to_images(pdf_path, dpi=Config.PDF_DPI)
            
            if not pages:
                return []
            
            invoices = []
            for page_num, page_image in enumerate(pages):
                logger.info(f"  Page {page_num + 1}/{len(pages)}")
                
                try:
                    invoice_data = self.extract_invoice_from_image(
                        page_image, 
                        f"{filename}_page_{page_num + 1}"
                    )
                    
                    invoice_data['invoice']['metadata'] = {
                        'source_file': filename,
                        'page_number': page_num + 1,
                        'total_pages': len(pages),
                        'file_type': 'pdf',
                        'processed_at': datetime.now().isoformat(),
                        'file_hash': self.get_file_hash(pdf_path)
                    }
                    
                    invoices.append(invoice_data)
                    self.processing_stats['successful'] += 1
                    
                except Exception as e:
                    logger.error(f"  Failed page {page_num + 1}: {e}")
                    self.processing_stats['failed'] += 1
            
            return invoices
            
        except Exception as e:
            logger.error(f"Failed PDF {pdf_path.name}: {e}")
            self.processing_stats['failed'] += 1
            return []
    
    def process_image(self, image_path: Path, filename: str) -> Optional[Dict]:
        """Process a single image file"""
        logger.info(f"Processing image: {image_path.name}")
        
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            invoice_data = self.extract_invoice_from_image(img, filename)
            
            invoice_data['invoice']['metadata'] = {
                'source_file': filename,
                'file_type': 'image',
                'processed_at': datetime.now().isoformat(),
                'file_hash': self.get_file_hash(image_path),
                'image_size': f"{img.width}x{img.height}"
            }
            
            self.processing_stats['successful'] += 1
            return invoice_data
            
        except Exception as e:
            logger.error(f"Failed image {image_path.name}: {e}")
            self.processing_stats['failed'] += 1
            return None
    
    def process_folder(self):
        """Main processing loop"""
        logger.info("=" * 70)
        logger.info("STARTING INVOICE PROCESSING")
        logger.info("=" * 70)
        logger.info(f"Input: {Config.INPUT_DIR}")
        logger.info(f"Model: {Config.MODEL_NAME}")
        
        self.processed_files = self.load_processed_files()
        logger.info(f"Already processed: {len(self.processed_files)} files")
        
        # Get files
        invoice_files = []
        for ext in Config.SUPPORTED_EXTENSIONS:
            invoice_files.extend(Config.INPUT_DIR.glob(f"*{ext}"))
            invoice_files.extend(Config.INPUT_DIR.glob(f"*{ext.upper()}"))
        
        invoice_files = list(set(invoice_files))
        invoice_files = [f for f in invoice_files if f.name not in self.processed_files]
        
        self.processing_stats['total_files'] = len(invoice_files)
        logger.info(f"Found {len(invoice_files)} new file(s)")
        
        if not invoice_files:
            logger.info("No new files to process!")
            return
        
        # Process files
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {}
            
            for file_path in invoice_files:
                if file_path.suffix.lower() == '.pdf':
                    future = executor.submit(self.process_pdf, file_path, file_path.stem)
                else:
                    future = executor.submit(self.process_image, file_path, file_path.stem)
                futures[future] = file_path
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    
                    if result:
                        if isinstance(result, list):
                            for page_result in result:
                                if page_result:
                                    self.invoices_data.append(page_result)
                        else:
                            self.invoices_data.append(result)
                        
                        self.processed_files.add(file_path.name)
                        processed_path = Config.PROCESSED_DIR / file_path.name
                        
                        if processed_path.exists():
                            base = processed_path.stem
                            ext = processed_path.suffix
                            counter = 1
                            while processed_path.exists():
                                processed_path = Config.PROCESSED_DIR / f"{base}_{counter}{ext}"
                                counter += 1
                        
                        file_path.rename(processed_path)
                        logger.info(f"Moved: {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed {file_path.name}: {e}")
        
        # Save results
        self.save_results()
        self.save_processed_files()
        self.generate_summary()
        self.print_statistics()
        
        logger.info("=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)
    
    def save_results(self):
        """Save JSON outputs"""
        for idx, invoice_data in enumerate(self.invoices_data):
            metadata = invoice_data['invoice'].get('metadata', {})
            filename = metadata.get('source_file', f'invoice_{idx+1}')
            
            clean_name = filename.replace('.', '_').replace(' ', '_')
            if 'page_number' in metadata:
                clean_name = f"{clean_name}_p{metadata['page_number']}"
            
            output_path = Config.OUTPUT_DATA_DIR / f"{clean_name}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(invoice_data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            logger.info(f"Saved: {output_path.name}")
        
        # Combined report
        combined_report = {
            "processing_info": {
                "processed_at": datetime.now().isoformat(),
                "total_invoices": len(self.invoices_data),
                "model_used": Config.MODEL_NAME,
                "statistics": {
                    "total_files": self.processing_stats['total_files'],
                    "successful": self.processing_stats['successful'],
                    "failed": self.processing_stats['failed'],
                    "start_time": self.processing_stats['start_time'].isoformat()
                }
            },
            "invoices": [data["invoice"] for data in self.invoices_data]
        }
        
        combined_path = Config.OUTPUT_REPORT_DIR / "all_invoices.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_report, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        logger.info(f"Combined report: {combined_path}")
    
    def generate_summary(self):
        """Generate CSV summary"""
        import csv
        
        summary_path = Config.OUTPUT_REPORT_DIR / "summary.csv"
        
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = [
                'source_file', 'page', 'invoice_number', 'status', 'issue_date', 
                'due_date', 'payment_date', 'vendor_name', 'vendor_city', 
                'customer_name', 'customer_city', 'subtotal', 'tax_amount', 
                'grand_total', 'amount_paid', 'balance_due', 'currency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in self.invoices_data:
                inv = data['invoice']
                metadata = inv.get('metadata', {})
                
                writer.writerow({
                    'source_file': metadata.get('source_file', 'unknown'),
                    'page': metadata.get('page_number', 1),
                    'invoice_number': inv.get('invoice_number', ''),
                    'status': inv.get('status', ''),
                    'issue_date': inv.get('issue_date', ''),
                    'due_date': inv.get('due_date', ''),
                    'payment_date': inv.get('payment_date', ''),
                    'vendor_name': inv.get('vendor', {}).get('company_name', ''),
                    'vendor_city': inv.get('vendor', {}).get('city', ''),
                    'customer_name': inv.get('customer', {}).get('company_name', ''),
                    'customer_city': inv.get('customer', {}).get('city', ''),
                    'subtotal': inv.get('totals', {}).get('subtotal', 0),
                    'tax_amount': inv.get('totals', {}).get('tax_amount', 0),
                    'grand_total': inv.get('totals', {}).get('grand_total', 0),
                    'amount_paid': inv.get('totals', {}).get('amount_paid', 0),
                    'balance_due': inv.get('totals', {}).get('balance_due', 0),
                    'currency': inv.get('currency', 'USD')
                })
        
        logger.info(f"CSV summary: {summary_path}")
    
    def print_statistics(self):
        """Print stats"""
        elapsed = (datetime.now() - self.processing_stats['start_time']).total_seconds()
        
        logger.info(f"\nTime: {elapsed:.2f}s")
        logger.info(f"Files: {self.processing_stats['total_files']}")
        logger.info(f"Success: {self.processing_stats['successful']}")
        logger.info(f"Failed: {self.processing_stats['failed']}")
        
        if self.invoices_data:
            total_value = sum(
                data['invoice'].get('totals', {}).get('grand_total', 0) 
                for data in self.invoices_data
            )
            logger.info(f"Total value: ${total_value:,.2f}")
            logger.info(f"Invoices: {len(self.invoices_data)}")

# ========== Main ==========
def main():
    try:
        processor = InvoiceExtractor()
        processor.process_folder()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()