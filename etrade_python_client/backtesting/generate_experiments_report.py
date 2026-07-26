import os
import json
from datetime import datetime

# We need the tracker to get the current engine hash
from backtesting.experiment_tracker import ExperimentTracker

_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments_log.jsonl"
)
_OUTPUT_HTML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments_dashboard.html"
)

def load_experiments(log_file=_LOG_FILE):
    if not os.path.exists(log_file):
        return []
        
    experiments = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                experiments.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return experiments

def generate_html_report(log_file=_LOG_FILE, output_file=_OUTPUT_HTML):
    experiments = load_experiments(log_file)
    tracker = ExperimentTracker()
    current_engine_hash = tracker._get_engine_hash()

    # CSS for a premium dark mode dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backtest Experiments Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-dark: #0f172a;
                --bg-card: #1e293b;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --accent-blue: #3b82f6;
                --accent-green: #10b981;
                --accent-red: #ef4444;
                --accent-warning: #f59e0b;
                --border-color: #334155;
            }}
            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-dark);
                color: var(--text-main);
                margin: 0;
                padding: 40px 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
            }}
            .header h1 {{
                margin: 0;
                font-weight: 700;
                font-size: 2.2rem;
            }}
            .engine-status {{
                background-color: var(--bg-card);
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 0.9rem;
                display: flex;
                align-items: center;
                gap: 10px;
                border: 1px solid var(--border-color);
            }}
            .hash-badge {{
                background-color: var(--accent-blue);
                color: white;
                padding: 3px 8px;
                border-radius: 4px;
                font-family: monospace;
                font-weight: 600;
            }}
            h2 {{
                font-weight: 600;
                margin-top: 40px;
                margin-bottom: 20px;
                color: var(--text-main);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background-color: var(--bg-card);
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }}
            th, td {{
                padding: 16px 20px;
                text-align: left;
                border-bottom: 1px solid var(--border-color);
            }}
            th {{
                background-color: #0f172a;
                color: var(--text-muted);
                font-weight: 600;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            tr:hover {{
                background-color: #273549;
            }}
            .positive {{ color: var(--accent-green); font-weight: 600; }}
            .negative {{ color: var(--accent-red); font-weight: 600; }}
            .warning-badge {{
                background-color: rgba(245, 158, 11, 0.2);
                color: var(--accent-warning);
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                display: inline-block;
                margin-top: 4px;
            }}
            .strategy-name {{
                font-weight: 700;
                font-size: 1.1rem;
            }}
            .strategy-link {{
                color: var(--accent-blue);
                text-decoration: none;
                transition: color 0.2s, opacity 0.2s;
            }}
            .strategy-link:hover {{
                color: #60a5fa;
                text-decoration: underline;
                opacity: 0.8;
            }}
            .date-range {{
                font-size: 0.85rem;
                color: var(--text-muted);
                margin-top: 2px;
            }}
            .empty-state {{
                text-align: center;
                padding: 60px 20px;
                background-color: var(--bg-card);
                border-radius: 10px;
                color: var(--text-muted);
            }}
            .action-btn {{
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 600;
                cursor: pointer;
                border: 1px solid var(--border-color);
                background: var(--bg-card);
                color: var(--text-main);
                transition: all 0.2s;
                margin-left: 5px;
            }}
            .btn-regen:hover {{ background-color: var(--accent-blue); border-color: var(--accent-blue); color: white; }}
            .btn-delete:hover {{ background-color: var(--accent-red); border-color: var(--accent-red); color: white; }}
            
            #status-bar {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                padding: 12px 24px;
                border-radius: 8px;
                background: #1e293b;
                color: white;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
                display: none;
                z-index: 1000;
                font-size: 0.9rem;
                border: 1px solid var(--accent-blue);
            }}
            
            .server-status {{
                font-size: 0.75rem;
                padding: 4px 8px;
                border-radius: 12px;
                background: #334155;
                color: #94a3b8;
                margin-left: 10px;
            }}
            .status-online {{ color: #10b981; }}
        </style>
        <script>
            function notify(msg, is_error=False) {{
                const bar = document.getElementById('status-bar');
                bar.innerText = msg;
                bar.style.display = 'block';
                bar.style.borderColor = is_error ? '#ef4444' : '#3b82f6';
                setTimeout(() => bar.style.display = 'none', 5000);
            }}

            async function deleteRun(id) {{
                if (!confirm('Are you sure you want to delete this experiment? This will remove the log and report file.')) return;
                try {{
                    const resp = await fetch(`http://localhost:8080/?id=${{id}}`, {{ method: 'DELETE' }});
                    const data = await resp.json();
                    if (resp.ok) {{
                        notify('Experiment deleted. Reloading...');
                        setTimeout(() => location.reload(), 1000);
                    }} else {{
                        notify('Error: ' + data.message, true);
                    }}
                }} catch (e) {{
                    notify('Manager server not running. Start it with: python backtesting/experiment_manager.py --serve', true);
                }}
            }}

            async function regenerateRun(id) {{
                notify('Regeneration started... this may take a minute.');
                try {{
                    const resp = await fetch(`http://localhost:8080/?id=${{id}}`, {{ method: 'POST' }});
                    const data = await resp.json();
                    if (resp.ok) {{
                        notify('Regeneration complete! Reloading...');
                        setTimeout(() => location.reload(), 1000);
                    }} else {{
                        notify('Error: ' + data.message, true);
                    }}
                }} catch (e) {{
                    notify('Manager server not running. Start it with: python backtesting/experiment_manager.py --serve', true);
                }}
            }}

            // Check server status
            async function checkServer() {{
                try {{
                    await fetch('http://localhost:8080/', {{ method: 'OPTIONS' }});
                    document.getElementById('server-badge').innerHTML = '<span class="status-online">●</span> Manager Online';
                }} catch (e) {{
                    document.getElementById('server-badge').innerHTML = '● Manager Offline (run experiment_manager.py --serve)';
                }}
            }}
            window.onload = checkServer;
        </script>
    </head>
    <body>
        <div id="status-bar"></div>
        <div class="container">
            <div class="header">
                <div>
                    <h1>🔬 Experiments Dashboard</h1>
                    <div id="server-badge" class="server-status">Checking manager status...</div>
                </div>
                <div class="engine-status">
                    Current Engine Hash: <span class="hash-badge">{current_engine_hash}</span>
                </div>
            </div>
    """

    if not experiments:
        html_content += """
            <div class="empty-state">
                <h2>No experiments recorded yet</h2>
                <p>Run a backtest using the backtest_runner.py to see your results here.</p>
            </div>
        </div>
        </body>
        </html>
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Generated empty dashboard at {output_file}")
        return

    # Sort experiments by start date (descending) for Recent Runs
    recent_runs = sorted(experiments, key=lambda x: x["timestamp"], reverse=True)

    # Leaderboard (group by strategy_hash to get the best iteration per strategy)
    best_runs = {}
    for exp in experiments:
        shash = exp["strategy_hash"]
        cagr = exp["metrics"].get("cagr_pct", 0)
        
        # Only consider runs that are not totally broken
        if cagr > -100:
            if shash not in best_runs or cagr > best_runs[shash]["metrics"].get("cagr_pct", 0):
                best_runs[shash] = exp

    leaderboard = sorted(list(best_runs.values()), key=lambda x: x["metrics"].get("cagr_pct", 0), reverse=True)

    html_content += """
            <h2>🏆 Strategy Leaderboard</h2>
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>CAGR %</th>
                        <th>Total Return %</th>
                        <th>Max Drawdown %</th>
                        <th>Sharpe</th>
                        <th>Calmar</th>
                        <th>Win Rate</th>
                        <th>Date Range</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    def format_num(val, suffix="", is_pct=False):
        if val is None: return "-"
        css_class = ""
        if is_pct or suffix in ("%",):
            if val > 0: css_class = "positive"
            elif val < 0: css_class = "negative"
        return f'<span class="{css_class}">{val:.2f}{suffix}</span>'

    for run in leaderboard:
        m = run["metrics"]
        cagr = m.get("cagr_pct", 0)
        tr = m.get("total_return_pct", 0)
        dd = m.get("max_drawdown_pct", 0)
        calmar = m.get("calmar_ratio", 0)
        sharpe = m.get("sharpe_ratio", 0)
        wr = m.get("win_rate_pct", 0)
        
        is_outdated = run["engine_hash"] != current_engine_hash
        status_html = f'<span class="warning-badge">Outdated Engine ({run["engine_hash"]})</span>' if is_outdated else '<span style="color:var(--accent-green);font-size:0.85rem;">Up to date</span>'

        strategy_name = run.get("strategy_id", "unknown")
        report_link = run.get("report_path", "")
        strategy_display = f'<a href="{report_link}" class="strategy-link">{strategy_name}</a>' if report_link else strategy_name

        html_content += f"""
                    <tr>
                        <td>
                            <div class="strategy-name">{strategy_display}</div>
                            <div class="date-range" style="font-family:monospace">Hash: {run.get("strategy_hash", "")}</div>
                        </td>
                        <td>{format_num(cagr, "%", True)}</td>
                        <td>{format_num(tr, "%", True)}</td>
                        <td>{format_num(dd, "%", False)}</td>
                        <td>{format_num(sharpe, "", True)}</td>
                        <td>{format_num(calmar, "", True)}</td>
                        <td>{format_num(wr, "%", False)}</td>
                        <td><div class="date-range">{run.get("start_date")} → {run.get("end_date")}</div></td>
                        <td>{status_html}</td>
                        <td style="text-align:right; white-space:nowrap;">
                            <button class="action-btn btn-regen" onclick="regenerateRun('{run.get('experiment_id')}')" title="Re-run with current engine">Regen</button>
                            <button class="action-btn btn-delete" onclick="deleteRun('{run.get('experiment_id')}')" title="Delete experiment data">Del</button>
                        </td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>

            <h2>🕒 Recent Runs</h2>
            <table>
                <thead>
                    <tr>
                        <th>Run Date</th>
                        <th>Strategy</th>
                        <th>CAGR / DD / Sharpe / Calmar</th>
                        <th>Trades (Win %)</th>
                        <th>Data Gaps</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """

    for run in recent_runs[:20]: # Show last 20
        m = run["metrics"]
        cagr = m.get("cagr_pct", 0)
        dd = m.get("max_drawdown_pct", 0)
        calmar = m.get("calmar_ratio", 0)
        sharpe = m.get("sharpe_ratio", 0)
        wr = m.get("win_rate_pct", 0)
        trades = m.get("total_trades", 0)
        gaps = m.get("data_gap_count", 0) + m.get("critical_gap_count", 0)
        
        run_dt = datetime.fromisoformat(run["timestamp"]).strftime("%Y-%m-%d %H:%M")
        
        is_outdated = run["engine_hash"] != current_engine_hash
        status_html = f'<span class="warning-badge">Outdated Engine</span>' if is_outdated else '<span style="color:var(--accent-green);font-size:0.85rem;">Current</span>'
        
        gaps_html = f'<span class="negative">{gaps}</span>' if gaps > 0 else f'<span class="text-muted">0</span>'

        strategy_name = run.get("strategy_id", "unknown")
        report_link = run.get("report_path", "")
        strategy_display = f'<a href="{report_link}" class="strategy-link">{strategy_name}</a>' if report_link else strategy_name

        html_content += f"""
                    <tr>
                        <td style="white-space:nowrap;"><div class="date-range">{run_dt}</div></td>
                        <td>
                            <div class="strategy-name">{strategy_display}</div>
                            <div class="date-range">{run.get("start_date")} → {run.get("end_date")}</div>
                        </td>
                        <td style="white-space:nowrap;">
                            {format_num(cagr, "%", True)} / {format_num(dd, "%")} / {format_num(sharpe, "", True)} / {format_num(calmar, "", True)}
                        </td>
                        <td>{trades} ({format_num(wr, "%")})</td>
                        <td>{gaps_html}</td>
                        <td>{status_html}</td>
                        <td style="text-align:right; white-space:nowrap;">
                            <button class="action-btn btn-regen" onclick="regenerateRun('{run.get('experiment_id')}')" title="Re-run with current engine">Regen</button>
                            <button class="action-btn btn-delete" onclick="deleteRun('{run.get('experiment_id')}')" title="Delete experiment data">Del</button>
                        </td>
                    </tr>
        """



    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Generated experiments dashboard at: {output_file}")


if __name__ == "__main__":
    generate_html_report()
