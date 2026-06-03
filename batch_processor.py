# batch_processor.py
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd

class BatchAnalyzer:
    """Analyze batch results across multiple runs"""
    
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.all_invoices = []
    
    def load_all_reports(self):
        """Load all combined reports"""
        for report_file in self.reports_dir.glob("all_invoices_*.json"):
            with open(report_file, 'r') as f:
                data = json.load(f)
                self.all_invoices.extend(data['invoices'])
        
        print(f"Loaded {len(self.all_invoices)} total invoices")
    
    def generate_dashboard(self):
        """Generate statistics dashboard"""
        df = pd.DataFrame([inv for inv in self.all_invoices if inv.get('totals')])
        
        if df.empty:
            print("No valid invoice data found")
            return
        
        # Calculate metrics
        stats = {
            'total_invoices': len(df),
            'total_value': df['totals'].apply(lambda x: x.get('grand_total', 0)).sum(),
            'average_invoice': df['totals'].apply(lambda x: x.get('grand_total', 0)).mean(),
            'paid_invoices': len(df[df.get('status') == 'Paid']),
            'pending_invoices': len(df[df.get('status') == 'Pending']),
            'unique_vendors': df['vendor'].apply(lambda x: x.get('company_name')).nunique(),
            'date_range': f"{df['issue_date'].min()} to {df['issue_date'].max()}"
        }
        
        # Save dashboard
        dashboard_path = self.reports_dir / "dashboard_stats.json"
        with open(dashboard_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n📊 Dashboard Statistics:")
        for key, value in stats.items():
            if 'value' in key or 'total' in key:
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value}")

# Usage
if __name__ == "__main__":
    analyzer = BatchAnalyzer(Path("output/combined_report"))
    analyzer.load_all_reports()
    analyzer.generate_dashboard()