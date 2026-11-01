"""Single CLI for generation, investigation and connectivity checks."""
import argparse
import json
import time
from pathlib import Path
from .workspace import ROOT, write_json


def show_progress(event):
    if event['event'] == 'model_started':
        print(f"Modèle : appel {event['call']}", flush=True)
    elif event['event'] == 'tool_completed':
        print(f"{event['evidence_id']} {event['tool']} : {event['result'].get('status')}", flush=True)


def main(argv=None):
    from .scenarios import SCENARIOS, generate_scenarios
    parser = argparse.ArgumentParser(description='Agent MLOps et scénarios synthétiques.')
    commands = parser.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('investigate', help='Investiguer un incident.')
    inv.add_argument('--scenario', choices=[*SCENARIOS, 'all'], default='concept_drift')
    inv.add_argument('--experiment', default='scenario_data/demo')
    inv.add_argument('--episode-dir', help='Incident externe ; référence fournie avec --experiment.')
    inv.add_argument('--provider', choices=['nvidia', 'local'], default='nvidia')
    inv.add_argument('--output', default='results')
    inv.add_argument('--evaluate', action='store_true', help='Comparer à la vérité synthétique après la fin uniquement.')
    gen = commands.add_parser('generate', help='Générer de nouveaux lots.')
    gen.add_argument('--scenario', choices=[*SCENARIOS, 'all'], default='all')
    gen.add_argument('--seed', type=int, default=100001)
    gen.add_argument('--reference', default='scenario_data/demo')
    gen.add_argument('--output', default='scenario_data/generated')
    gen.add_argument('--label', default='generated')
    check = commands.add_parser('check', help='Une seule question pour mesurer la réponse du modèle.')
    check.add_argument('--provider', choices=['nvidia', 'local'], default='nvidia')
    check.add_argument('--question', default='Combien font 17 × 23 ? Réponds brièvement en français.')
    args = parser.parse_args(argv)
    if args.command == 'generate':
        generate_scenarios(args.scenario, args.seed, args.reference, args.output, args.label)
        return 0
    if args.command == 'check':
        from .models import load_local_key, make_model, message_text
        filename = 'nemotron.json' if args.provider == 'nvidia' else 'local.json'
        config = json.loads((ROOT / 'configs' / filename).read_text(encoding='utf-8'))
        if args.provider == 'nvidia':
            load_local_key()
            config.update(max_output_tokens=256, reasoning=False)
        else:
            config.update(num_predict=256, reasoning=False)
        started = time.monotonic()
        response = make_model(config).invoke(args.question)
        print(json.dumps({'response': message_text(response), 'duration_seconds': time.monotonic() - started,
                          'usage': response.usage_metadata}, ensure_ascii=False, indent=2))
        return 0
    from .agent import investigate
    if args.episode_dir and args.scenario == 'all':
        parser.error('--episode-dir est incompatible avec --scenario all.')
    episodes = [args.episode_dir] if args.episode_dir else [str(ROOT / args.experiment / 'episodes' / SCENARIOS[s])
        for s in (list(SCENARIOS) if args.scenario == 'all' else [args.scenario])]
    failed = False
    for episode in episodes:
        result = investigate(episode, args.experiment, provider=args.provider,
                             output_dir=args.output, on_event=show_progress)
        if args.evaluate:
            truth = json.loads((ROOT / args.experiment / 'evaluation/ground_truth.json').read_text(encoding='utf-8'))
            expected = next(t['expected_cause'] for t in truth if t['episode_id'] == Path(episode).name)
            evaluation = {'expected_cause': expected,
                          'cause_correct': (result['final_diagnosis'] or {}).get('cause_probable') == expected}
            write_json(Path(result['run_directory']) / 'evaluation.json', evaluation)
            print(json.dumps(evaluation, ensure_ascii=False))
        print(json.dumps({k: v for k, v in result.items() if k != 'evidence'}, ensure_ascii=False, indent=2))
        failed |= result['status'] != 'completed'
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
