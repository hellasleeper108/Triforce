from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Input, DataTable, Log, Digits
from textual.reactive import reactive
from textual.binding import Binding
import random
import asyncio
import datetime

# Mock Data Generator
def get_node_stats():
    return [
        ("odin", "Master", random.randint(10, 30), random.randint(40, 50), 0),
        ("thor", "Worker", random.randint(70, 95), random.randint(80, 95), random.randint(80, 100)),
        ("loki", "Worker", random.randint(5, 15), random.randint(20, 30), random.randint(0, 10)),
    ]

class NodePanel(Static):
    """Panel showing node metrics."""
    def compose(self) -> ComposeResult:
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Node", "Role", "CPU%", "RAM%", "GPU%")
        self.update_table()
        self.set_interval(1, self.update_table)

    def update_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        stats = get_node_stats()
        for s in stats:
            # Color coding for high load
            cpu = f"[red]{s[2]}%[/red]" if s[2] > 80 else f"{s[2]}%"
            ram = f"[red]{s[3]}%[/red]" if s[3] > 80 else f"{s[3]}%"
            gpu = f"[red]{s[4]}%[/red]" if s[4] > 80 else f"{s[4]}%"
            table.add_row(s[0], s[1], cpu, ram, gpu)

class AlertPanel(Static):
    """Scrolling log of alerts."""
    
    def compose(self) -> ComposeResult:
        yield Log()

    def on_mount(self) -> None:
        log = self.query_one(Log)
        log.write("[green]System Online[/green]")
        self.set_interval(3, self.add_simulated_alert)

    def add_simulated_alert(self) -> None:
        log = self.query_one(Log)
        events = [
            ("INFO", "Heartbeat received from Thor"),
            ("WARN", "High memory usage on Thor (92%)"),
            ("INFO", "Task job-123 completed"),
            ("ERROR", "Connection timeout to Loki"),
            ("INFO", "Autoscaler spawned new agent"),
        ]
        lvl, msg = random.choice(events)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        color = "white"
        if lvl == "WARN": color = "yellow"
        if lvl == "ERROR": color = "red"
        
        log.write(f"[{color}]{timestamp} [{lvl}] {msg}[/{color}]")

class CommandPanel(Static):
    """Input for NL commands."""
    
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Ask STAN to do something...")

    def on_input_submitted(self, message: Input.Submitted) -> None:
        log = self.app.query_one("AlertPanel", AlertPanel).query_one(Log)
        log.write(f"\n[bold cyan]> {message.value}[/bold cyan]")
        
        # Simulate processing
        log.write("[italic]STAN is thinking...[/italic]")
        # In a real app, this would call STANSupernova().act(message.value)
        # For TUI demo, we mock the response delay
        self.set_timer(1, lambda: log.write(f"[bold green]OK. Executing: {message.value}[/bold green]\n"))
        
        self.query_one(Input).value = ""

class STANApp(App):
    """The STAN HUD Application."""
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 2fr 1fr;
        grid-rows: 3fr 1fr;
    }
    
    NodePanel {
        row-span: 1;
        col-span: 1;
        border: solid green;
        background: $surface;
    }
    
    AlertPanel {
        row-span: 1;
        col-span: 1;
        border: solid red;
        background: $surface;
    }
    
    CommandPanel {
        col-span: 2;
        border: solid blue;
        height: 3;
        dock: bottom;
    }
    
    Header {
        dock: top;
    }
    """
    
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield NodePanel()
        yield AlertPanel()
        yield CommandPanel()
        yield Footer()

if __name__ == "__main__":
    app = STANApp()
    app.run()
