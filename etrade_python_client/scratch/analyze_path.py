import json
import sys

def analyze_log(log_path):
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    path = data.get('path', [])
    for day in path:
        date = day['date']
        events = day.get('events', [])
        active = day.get('active_trades', [])
        nlv = day.get('nlv')
        margin = day.get('margin', 0)
        regime = day.get('regime_name')
        
        event_types = [e['type'] for e in events]
        event_str = f"Events: {event_types}" if event_types else ""
        
        print(f"{date} | {regime[:15]:15} | NLV: {nlv:7.0f} | Margin: {margin:6.0f} | Active: {len(active)} | {event_str}")

if __name__ == "__main__":
    analyze_log(sys.argv[1])
