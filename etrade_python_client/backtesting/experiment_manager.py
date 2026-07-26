import os
import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

LOG_FILE = "backtesting/experiments_log.jsonl"
REPORTS_DIR = "backtesting/reports"

class ExperimentManager:
    def delete_experiment(self, experiment_id):
        """Removes an experiment from the log and deletes its report file."""
        if not os.path.exists(LOG_FILE):
            return False, "Log file not found."

        temp_log = LOG_FILE + ".tmp"
        deleted = False
        report_to_delete = None
        
        with open(LOG_FILE, "r") as f, open(temp_log, "w") as out:
            for line in f:
                data = json.loads(line)
                if data.get("experiment_id") == experiment_id:
                    deleted = True
                    report_to_delete = data.get("report_path")
                    continue
                out.write(line)
        
        if deleted:
            os.replace(temp_log, LOG_FILE)
            # Delete report file
            if report_to_delete:
                full_report_path = os.path.join("backtesting", report_to_delete)
                if os.path.exists(full_report_path):
                    os.remove(full_report_path)
            return True, f"Experiment {experiment_id} deleted."
        else:
            os.remove(temp_log)
            return False, f"Experiment {experiment_id} not found."

    def regenerate_experiment(self, experiment_id):
        """Re-runs a strategy with the same parameters using the current engine."""
        if not os.path.exists(LOG_FILE):
            return False, "Log file not found."

        target_run = None
        with open(LOG_FILE, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("experiment_id") == experiment_id:
                    target_run = data
                    break
        
        if not target_run:
            return False, f"Experiment {experiment_id} not found."

        config = target_run["strategy_config"]
        start_date = target_run["start_date"]
        end_date = target_run["end_date"]
        
        # Save temporary config for re-run
        temp_config_path = f"backtesting/temp_config_{experiment_id}.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f)

        try:
            cmd = [
                sys.executable, "backtesting/backtest_runner.py",
                "--strategy_config", temp_config_path,
                "--start", start_date,
                "--end", end_date
            ]
            print(f"Regenerating: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Update the dashboard report after generation
            subprocess.run([sys.executable, "backtesting/generate_experiments_report.py"])
            
            return True, f"Experiment {experiment_id} regenerated."
        except Exception as e:
            return False, f"Regeneration failed: {e}"
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

class DashboardHandler(BaseHTTPRequestHandler):
    manager = ExperimentManager()

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_DELETE(self):
        parsed_path = urlparse(self.path)
        qs = parse_qs(parsed_path.query)
        exp_id = qs.get("id", [None])[0]
        
        if exp_id:
            success, msg = self.manager.delete_experiment(exp_id)
            # Regenerate dashboard to reflect changes
            subprocess.run([sys.executable, "backtesting/generate_experiments_report.py"])
            self._set_headers(200 if success else 404)
            self.wfile.write(json.dumps({"message": msg}).encode())
        else:
            self._set_headers(400)
            self.wfile.write(json.dumps({"message": "Missing ID"}).encode())

    def do_POST(self):
        parsed_path = urlparse(self.path)
        qs = parse_qs(parsed_path.query)
        exp_id = qs.get("id", [None])[0]
        
        if exp_id:
            success, msg = self.manager.regenerate_experiment(exp_id)
            self._set_headers(200 if success else 500)
            self.wfile.write(json.dumps({"message": msg}).encode())
        else:
            self._set_headers(400)
            self.wfile.write(json.dumps({"message": "Missing ID"}).encode())

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"Experiment Manager Server running on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", help="Delete experiment by ID")
    parser.add_argument("--regenerate", help="Regenerate experiment by ID")
    parser.add_argument("--serve", action="store_true", help="Start the management server")
    args = parser.parse_args()

    mgr = ExperimentManager()
    if args.delete:
        s, m = mgr.delete_experiment(args.delete)
        print(m)
        subprocess.run([sys.executable, "backtesting/generate_experiments_report.py"])
    elif args.regenerate:
        s, m = mgr.regenerate_experiment(args.regenerate)
        print(m)
    elif args.serve:
        run_server()
    else:
        parser.print_help()
