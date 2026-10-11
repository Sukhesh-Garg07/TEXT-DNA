
from typing import Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

console = Console()

# REPORT PRINTER

def print_report(l1: Dict, l2: Dict, l3: Dict, l4: Dict, l5: Dict) -> None:

    console.print()
    console.print(Panel.fit(
        "[bold bright_white]  TextDNA — Full-Spectrum NLP Intelligence Report  [/bold bright_white]",
        border_style="bright_blue",
    ))

    # Layer 1 
    console.print()
    console.print(Rule("[bold yellow]Layer 1 — Linguistic Foundation[/bold yellow]"))

    t = Table(show_header=False, box=None, padding=(0, 3))
    t.add_column("K", style="cyan", no_wrap=True)
    t.add_column("V", style="white")
    t.add_row("Words",str(l1["total_words"]))
    t.add_row("Unique words", str(l1["unique_words"]))
    t.add_row("Sentences",str(l1["total_sentences"]))
    t.add_row("Avg sentence length",    f"{l1['avg_sentence_length']} words")
    t.add_row("Type-Token Ratio", str(l1["type_token_ratio"]) + "  (0–1, higher = richer vocab)")
    t.add_row("Avg dependency depth",  str(l1["avg_dependency_tree_depth"]) + "  (higher = more complex syntax)")
    t.add_row("Reading grade", l1["readability"]["grade_level"])
    t.add_row("Flesch ease",  str(l1["readability"]["flesch_reading_ease"]) + "  (0=hard, 100=easy)")
    t.add_row("Gunning Fog index", str(l1["readability"]["gunning_fog"]))
    console.print(t)

    if l1["named_entities"]:
        console.print("[cyan]Named Entities Detected:[/cyan]")
        for ent, label in l1["named_entities"][:10]:
            console.print(f"    [green]·[/green] {ent}  [dim]({label})[/dim]")

    # Layer 2 
    console.print()
    console.print(Rule("[bold yellow]Layer 2 — Semantic Intelligence[/bold yellow]"))

    t2 = Table(show_header=False, box=None, padding=(0, 3))
    t2.add_column("K", style="cyan", no_wrap=True)
    t2.add_column("V", style="white")
    t2.add_row("Avg semantic coherence",  str(l2["avg_semantic_coherence"]) + "  (consecutive-sentence similarity)")
    t2.add_row("Semantic diversity", str(l2["semantic_diversity_score"]) + "  (how varied the topics are)")
    t2.add_row("Key concepts",  ", ".join(l2["key_concepts"][:8]))
    central = l2["central_sentence"]
    t2.add_row("Central sentence",  (central[:90] + "…") if len(central) > 90 else central)
    console.print(t2)

    if l2["semantic_drift_points"]:
        console.print("\n  [cyan]Semantic Drift Points (sudden topic shifts):[/cyan]")
        for d in l2["semantic_drift_points"][:4]:
            preview = d["sentence"][:70]
            console.print(f"    [red]·[/red] After sentence {d['position']}: [dim]\"{preview}…\"[/dim]")

    # Layer 3
    console.print()
    console.print(Rule("[bold yellow]Layer 3 — Argument Mining & Discourse[/bold yellow]"))

    s = l3["argument_structure_summary"]
    console.print(f"  Claims: [bold]{s['claims']}[/bold]   Evidence units: [bold]{s['evidence']}[/bold]   Conclusions: [bold]{s['conclusions']}[/bold]")

    console.print("\n  [cyan]Logical Connective Usage:[/cyan]")
    for ctype, count in l3["connective_analysis"].items():
        bar = "█" * min(count, 20)
        console.print(f"    {ctype:<14}  [green]{bar}[/green] ({count})")

    if l3["rhetorical_devices"]:
        console.print("\n  [cyan]Rhetorical Devices:[/cyan]")
        for d in l3["rhetorical_devices"]:
            console.print(f"    [magenta]·[/magenta] [bold]{d['device']}[/bold]: [dim]{d['sentence'][:70]}[/dim]")

    if l3["logical_fallacies"]:
        console.print("\n  [cyan]Logical Fallacy Signals:[/cyan]")
        for f in l3["logical_fallacies"]:
            console.print(f"    [red]⚠[/red]  [bold]{f['fallacy']}[/bold]  (conf: {f['confidence']})")
            console.print(f"       [dim]\"{f['sentence'][:80]}\"[/dim]")
    else:
        console.print("\n  [green]  No strong logical fallacy signals detected.[/green]")

    # Sample argument roles
    if l3["argument_roles"]:
        console.print("\n  [cyan]Sample Sentence Classifications:[/cyan]")
        for a in l3["argument_roles"][:5]:
            console.print(f"    [{_role_color(a['role'])}]{a['role'].upper()}[/{_role_color(a['role'])}] ({a['confidence']})  {a['sentence'][:70]}")

    # Layer 4
    console.print()
    console.print(Rule("[bold yellow]Layer 4 — Cognitive & Psychological Analysis[/bold yellow]"))

    mft = l4["moral_foundation_analysis"]
    console.print(f"  [cyan]Dominant Moral Frame:[/cyan]  [bold green]{mft['dominant_frame']}[/bold green]")
    console.print(f"  [cyan]Framing Type:[/cyan]  {l4['framing_analysis']['type']}")
    console.print(f"  [cyan]Agency:[/cyan]  {l4['agency_analysis']['note']}")

    console.print("\n  [cyan]Moral Foundation Distribution:[/cyan]")
    for foundation, pct in mft["percentages"].items():
        bar = "█" * int(pct / 4)
        console.print(f"    {foundation:<28}  [blue]{bar}[/blue] ({pct}%)")

    if l4["cognitive_bias_signals"]:
        console.print("[cyan]Cognitive Bias Signals:[/cyan]")
        for seg in l4["cognitive_bias_signals"][:3]:
            for bias, score in seg["detected_biases"]:
                console.print(f"    [yellow]⚠[/yellow]  {bias}  ({score})")
    else:
        console.print("[green]  No strong cognitive bias signals detected.[/green]")

    # Layer 5 
    console.print()
    console.print(Rule("[bold yellow]Layer 5 — Generative Intelligence[/bold yellow]"))

    score = l5["text_dna_score"]
    color = "green" if score >= 70 else "yellow" if score >= 40 else "red"
    console.print(f"[bold]TextDNA Score:  [{color}]{score} / 100[/{color}][/bold]")

    for component, value in l5["text_dna_breakdown"].items():
        bar   = "█" * int(value)
        label = component.replace("_", " ").title()
        console.print(f"    {label:<26}  [{color}]{bar}[/{color}]  {value}/25")

    if l5["signature_traits"]:
        console.print("[cyan]Signature Traits:[/cyan]")
        for trait in l5["signature_traits"]:
            console.print(f"    [green]✦[/green]  {trait}")

    if l5["implicit_assumptions"]:
        console.print("[cyan]Implicit Assumptions Detected:[/cyan]")
        for a in l5["implicit_assumptions"]:
            console.print(f"    [blue]·[/blue]  {a['assumption']}  ({a['confidence']})")

    if l5["information_gaps"]:
        console.print("[cyan]Information Gaps (questions the text raises but doesn't answer):[/cyan]")
        for g in l5["information_gaps"]:
            console.print(f"[yellow]?[/yellow]  {g['gap']}  ({g['relevance']})")

    console.print()
    console.print(Panel.fit(
        "[bold green]  ✓  TextDNA analysis complete.  Report saved → textdna_report.json  [/bold green]",
        border_style="green",
    ))
    console.print()


def _role_color(role: str) -> str:
    return {
        "claim" : "magenta",
        "supporting evidence": "green",
        "conclusion": "yellow",
        "counterargument": "red",
        "background information": "blue",
        "example" : "cyan",
    }.get(role, "white")
