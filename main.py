# invoice_processor.py
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import google.generativeai as genai
from PIL import Image
from pdf2image import convert_from_path
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Import configuration
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'processing_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== Define Your Schema ==========
class Vendor(BaseModel):
    company_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class Customer(BaseModel):
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None

class LineItem(BaseModel):
    item_id: Optional[str] = None
    description: str
    quantity: float
    unit_price: float
    discount: float = 0.0
    taxable: bool = True
    total_price: float

class Totals(BaseModel):
    subtotal: float
    tax_rate_percentage: float = 0.0
    tax_amount: float = 0.0
    discount_total: float = 0.0
    grand_total: float
    amount_paid: float = 0.0
    balance_due: float = 0.0

class PaymentDetails(BaseModel):
    method: Optional[str] = None
    transaction_reference: Optional[str] = None
    terms: Optional[str] = None

class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    status: Optional[str] = None  # "Paid", "Pending", "Overdue"
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
    
    class Config:
        extra = "allow"  # Allow extra fields not defined in schema

# ========== Gemini Setup ==========
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "response_mime_type": "application/json",
        "temperature": TEMPERATURE
    }
)

# ========== Invoice Extractor Class ==========
class InvoiceExtractor:
    def __init__(self):
        self.processed_files = set()
        self.invoices_data = []
        
    def load_processed_files(self) -> set:
        """Load previously processed files to avoid reprocessing"""
        processed_file = OUTPUT_REPORT_DIR / "processed_files.json"
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_files(self):
        """Save list of processed files"""
        processed_file = OUTPUT_REPORT_DIR / "processed_files.json"
        with open(processed_file, 'w') as f:
            json.dump(list(self.processed_files), f, indent=2)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file for duplicate detection"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract_invoice_from_image(self, image: Image.Image, filename: str) -> Dict:
        """Extract invoice data from a single image using Gemini"""
        
        prompt = """
        Extract invoice data from this image according to the exact JSON schema provided.
        
        Important rules:
        1. For any field you cannot find, use null (not empty strings)
        2. Convert dates to YYYY-MM-DD format
        3. Ensure all monetary values are numbers (not strings)
        4. Calculate totals if they're not explicitly shown:
           - subtotal = sum of line_item.total_price
           - grand_total = subtotal + tax_amount - discount_total
           - balance_due = grand_total - amount_paid
        5. For line_items, calculate total_price = quantity * unit_price - discount
        6. If tax rate is shown, calculate tax_amount = subtotal * (tax_rate/100)
        
        Return ONLY valid JSON matching this structure:
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
        """
        
        response = model.generate_content([prompt, image])
        
        # Parse and validate response
        try:
            data = json.loads(response.text)
            # Ensure it has the invoice wrapper
            if "invoice" not in data:
                data = {"invoice": data}
            
            # Validate against Pydantic model
            validated = Invoice(**data["invoice"])
            return data
            
        except Exception as e:
            logger.error(f"Failed to validate invoice data: {e}")
            logger.error(f"Raw response: {response.text}")
            raise
    
    def process_pdf(self, pdf_path: Path, filename: str) -> List[Dict]:
        """Convert PDF to images and process each page"""
        logger.info(f"Converting PDF: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=PDF_DPI)
        
        invoices = []
        for page_num, page in enumerate(pages):
            logger.info(f"Processing page {page_num + 1} of {len(pages)}")
            try:
                invoice_data = self.extract_invoice_from_image(page, f"{filename}_page_{page_num}")
                invoice_data['invoice']['_metadata'] = {
                    'source_file': filename,
                    'page_number': page_num + 1,
                    'total_pages': len(pages)
                }
                invoices.append(invoice_data)
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {e}")
                invoices.append({
                    'invoice': {
                        '_metadata': {
                            'source_file': filename,
                            'page_number': page_num + 1,
                            'error': str(e)
                        }
                    }
                })
        
        return invoices
    
    def process_image(self, image_path: Path, filename: str) -> Dict:
        """Process a single image file"""
        logger.info(f"Processing image: {image_path}")
        img = Image.open(image_path)
        
        # Optional: Resize large images
        if img.width > 2000:
            ratio = 2000 / img.width
            new_size = (2000, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        invoice_data = self.extract_invoice_from_image(img, filename)
        invoice_data['invoice']['_metadata'] = {
            'source_file': filename,
            'processed_at': datetime.now().isoformat(),
            'file_hash': self.get_file_hash(image_path)
        }
        
        return invoice_data
    
    def process_folder(self):
        """Main processing loop for the entire folder"""
        logger.info("=" * 60)
        logger.info("Starting Invoice Processing")
        logger.info("=" * 60)
        
        # Load already processed files
        self.processed_files = self.load_processed_files()
        
        # Get all invoice files
        invoice_files = []
        for ext in SUPPORTED_EXTENSIONS:
            invoice_files.extend(INPUT_DIR.glob(f"*{ext}"))
            invoice_files.extend(INPUT_DIR.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(invoice_files)} invoice files")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            
            for file_path in invoice_files:
                if file_path.name in self.processed_files:
                    logger.info(f"Skipping already processed: {file_path.name}")
                    continue
                
                if file_path.suffix.lower() == '.pdf':
                    future = executor.submit(self.process_pdf, file_path, file_path.stem)
                else:
                    future = executor.submit(self.process_image, file_path, file_path.stem)
                
                futures[future] = file_path
            
            # Collect results
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    
                    if isinstance(result, list):
                        # PDF with multiple pages
                        for page_result in result:
                            self.invoices_data.append(page_result)
                    else:
                        # Single image
                        self.invoices_data.append(result)
                    
                    # Mark as processed and move file
                    self.processed_files.add(file_path.name)
                    processed_path = PROCESSED_DIR / file_path.name
                    file_path.rename(processed_path)
                    logger.info(f"Successfully processed: {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
        
        # Save results
        self.save_results()
        self.save_processed_files()
        self.generate_summary()
        
        logger.info("=" * 60)
        logger.info("Processing Complete!")
        logger.info(f"Processed {len(self.invoices_data)} invoices")
        logger.info("=" * 60)
    
    def save_results(self):
        """Save individual and combined JSON outputs"""
        
        # Save individual JSON files
        for idx, invoice_data in enumerate(self.invoices_data):
            if '_metadata' in invoice_data['invoice']:
                filename = invoice_data['invoice']['_metadata'].get('source_file', f'invoice_{idx}')
            else:
                filename = f'invoice_{idx}'
            
            # Clean filename for saving
            clean_name = filename.replace('.', '_').replace(' ', '_')
            output_path = OUTPUT_DATA_DIR / f"{clean_name}.json"
            
            with open(output_path, 'w') as f:
                json.dump(invoice_data, f, indent=2)
            logger.info(f"Saved: {output_path}")
        
        # Save combined report
        combined_report = {
            "processing_info": {
                "processed_at": datetime.now().isoformat(),
                "total_invoices": len(self.invoices_data),
                "model_used": MODEL_NAME
            },
            "invoices": [data["invoice"] for data in self.invoices_data]
        }
        
        combined_path = OUTPUT_REPORT_DIR / "all_invoices.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_report, f, indent=2)
        logger.info(f"Combined report saved: {combined_path}")
    
    def generate_summary(self):
        """Generate CSV summary of all invoices"""
        import csv
        
        summary_path = OUTPUT_REPORT_DIR / "summary.csv"
        
        with open(summary_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'source_file', 'invoice_number', 'status', 'issue_date', 
                'due_date', 'payment_date', 'vendor_name', 'customer_name',
                'subtotal', 'tax_amount', 'grand_total', 'amount_paid', 
                'balance_due', 'currency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in self.invoices_data:
                inv = data['invoice']
                metadata = inv.get('_metadata', {})
                
                writer.writerow({
                    'source_file': metadata.get('source_file', 'unknown'),
                    'invoice_number': inv.get('invoice_number', ''),
                    'status': inv.get('status', ''),
                    'issue_date': inv.get('issue_date', ''),
                    'due_date': inv.get('due_date', ''),
                    'payment_date': inv.get('payment_date', ''),
                    'vendor_name': inv.get('vendor', {}).get('company_name', ''),
                    'customer_name': inv.get('customer', {}).get('company_name', ''),
                    'subtotal': inv.get('totals', {}).get('subtotal', 0),
                    'tax_amount': inv.get('totals', {}).get('tax_amount', 0),
                    'grand_total': inv.get('totals', {}).get('grand_total', 0),
                    'amount_paid': inv.get('totals', {}).get('amount_paid', 0),
                    'balance_due': inv.get('totals', {}).get('balance_due', 0),
                    'currency': inv.get('currency', 'USD')
                })
        
        logger.info(f"Summary CSV saved: {summary_path}")

# ========== Main Execution ==========
def main():
    # Check API key
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        logger.error("Please set your GEMINI_API_KEY in config.py or environment variable")
        sys.exit(1)
    
    # Create processor and run
    processor = InvoiceExtractor()
    processor.process_folder()

if __name__ == "__main__":
    main()