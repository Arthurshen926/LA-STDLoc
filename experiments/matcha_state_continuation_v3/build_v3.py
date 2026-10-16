"""Build an isolated state-exact continuation version; never edit frozen v1/v2."""
from pathlib import Path
import ast, difflib, hashlib, json
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "matcha_exact_resume_v2"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert sha(OLD / "exact_resume.py") == "3a95c8747b3a1e3a96494edec4d40e96afa31edbfbc978684f16291655d89c03"
    names = ("diagnostic_cpu.py", "native_n8_launcher.py", "continuation_gate.py",
             "project_a_reference.py", "freeze_binding.py", "build_trainer.py", "build_observer.py",
             "train_with_charts_exact.py", "train_with_charts_observer.py")
    copied = {}
    for name in names:
        p = ROOT / name
        assert not p.exists()
        p.write_bytes((OLD / name).read_bytes())
        copied[name] = sha(p)
    original = (OLD / "exact_resume.py").read_text()
    text = original
    def replace(a,b):
        nonlocal text
        assert text.count(a) == 1, a[:100]
        text = text.replace(a,b)
    replace("Candidate engineering extension; GPU equivalence and production use are gated.",
            "State-exact checkpoint transport only. Native floating trajectory equality is NOT claimed.")
    replace("import resource\n", "import resource\nimport math\n")
    replace('    return {"dtype": type(value).__name__, "hex": float(value).hex()}',
            '    require(math.isfinite(float(value)), "nonfinite Python loss component")\n'
            '    return {"dtype": type(value).__name__, "hex": float(value).hex()}')
    a=text.index("def validate_reference("); b=text.index("def add_arguments(",a)
    text=text[:a]+text[b:]
    a=text.index('        gate_path = getattr(args, "exact_continue_gate", None)')
    b=text.index('        self.trace = ',a)
    text=text[:a]+'''        require(not args.exact_windows and not args.exact_reference_trace and
                not args.exact_continue_gate and not args.exact_reference_terminal and
                not args.exact_prefix_spec, "v3 has no trajectory comparison or promotion gate")
        if self.has_resume:
            require(self.state["completed_iteration"] == 14999 and self.state["next_iteration"] == 15000 and
                    args.exact_stop == 30000 and args.exact_save == [], "only14999->30000 resume")
        else:
            require(args.exact_stop == 14999 and args.exact_save == [14999], "only fresh1->14999")
        self.windows, self.reference = [], {}
        self.next_expected = 15000 if self.has_resume else 1
        self.restore_verified = not self.has_resume
        self.in_step = False
''' + text[b:]
    a=text.index("    def check_reference("); b=text.index("    def metric(",a)
    text=text[:a]+'''    def check_reference(self, record, key):
        require(not self.reference, "trajectory reference forbidden in v3")

''' + text[b:]
    replace('        require(equal(before, rng_state()), "restored verification consumed RNG")',
            '''        require(equal(before, rng_state()), "restored verification consumed RNG")
        self.restore_verified = True
        proof = {"schema": "matcha_state_restore_proof_v3", "verdict": "PASS",
                 "completed_iteration": 14999, "next_iteration": 15000,
                 "saved_full_digest": self.restored_expected_digest, "restored_full_digest": digest,
                 "static_digest": cache_digest, "all_state_bitwise": True,
                 "floating_trajectory_equivalence_claimed": False}
        with (self.root / "restore_bitwise_PASS.json").open("x") as f:
            json.dump(proof, f, sort_keys=True); f.flush(); os.fsync(f.fileno())''')
    a=text.index("    def before_step("); b=text.index("    def observe_view(",a)
    text=text[:a]+'''    def before_step(self, iteration):
        require(self.restore_verified and not self.in_step and iteration == self.next_expected,
                "unverified restore, skipped or repeated step")
        require(torch.get_num_threads() == self.binding["runtime"]["threads"] and
                torch.get_num_interop_threads() == self.binding["runtime"]["interop"],
                "diagnostic thread settings leaked into training")
        self.in_step = True

''' + text[b:]
    replace('        k = local["iteration"]\n',
            '        k = local["iteration"]\n'
            '        require(self.in_step and k == self.next_expected, "boundary order")\n')
    a=text.index('        if self.prefix_gate is not None and k <= 8:'); b=text.index('        if k == self.args.exact_stop:',a)
    text=text[:a]+'''        self.in_step = False
        self.next_expected = k + 1
''' + text[b:]
    replace('            validate_checkpoint(state, self.binding, self.static)',
            '            validate_checkpoint(state, self.binding, self.static)\n'
            '            finite_state((model, optimizer, state["loop"]))')
    replace('        if k == self.args.exact_stop:\n            self.metric',
            '        if k == self.args.exact_stop:\n'
            '            if k == 30000:\n'
            '                finite_terminal_model(local["gaussians"])\n'
            '            self.metric')
    insert='''
@diagnostic_only
def finite_state(value):
    """Finite screen only; no normalization, mutation or floating tolerance."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), "nonfinite saved tensor")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact):
            require(bool(np.isfinite(value).all()), "nonfinite saved array")
    elif isinstance(value, dict):
        for item in value.values(): finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite_state(item)
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(float(value)), "nonfinite saved scalar")

@diagnostic_only
def finite_terminal_model(model):
    # Original step30000 leaves gradients; do not call capture_model/zero_grad/step here.
    finite_state([getattr(model, name) for name in PARAMS.values()])
    finite_state(model.optimizer.state_dict())
    finite_state([getattr(model, name) for name in MODEL_FIELDS])
    finite_state([p.grad for g in model.optimizer.param_groups for p in g["params"]])

'''
    text=text.replace("class Controller:",insert+"class Controller:")
    ast.parse(text)
    assert not (ROOT/"exact_resume.py").exists()
    (ROOT/"exact_resume.py").write_text(text)
    (ROOT/"v2_to_v3.diff").write_text("".join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(OLD/"exact_resume.py"),tofile=str(ROOT/"exact_resume.py"))))
    def funcs(s):
        a=ast.parse(s)
        return {n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before, after=funcs(original),funcs(text)
    audit={"schema":"matcha_v3_ast_audit","copied_unchanged":copied,
      "changed_top_level":[k for k in before.keys() & after.keys() if before[k]!=after[k]],
      "removed_top_level":sorted(before.keys()-after.keys()),"added_top_level":sorted(after.keys()-before.keys()),
      "native_math_trainers_byte_identical_to_v2":True,"old_source_unchanged":sha(OLD/"exact_resume.py")}
    (ROOT/"ast_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True))
    print(json.dumps(audit,sort_keys=True))
if __name__=="__main__": main()
