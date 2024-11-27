# foldrpp_wrapper.py

from foldrpp import Foldrpp
import clingo

def train_foldrpp(model, data_train):
    model.fit(data_train)
    model.asp()
    return model

def model_to_asp(model, scale_factor=10):
    asp_rules = []
    operator_map = {
        '==': '=',
        '!=': '!=',
        '=<': '<=',
        '>=': '>=',
        '>': '>',
        '<': '<'
    }
    for rule in model.flat_rules:
        head = rule[0]
        main_items = rule[1]
        ab_items = rule[2]

        head_pred = f'{head[0]}(X, "{head[2]}")'
        body_preds = []
        for idx, item in enumerate(main_items):
            attr, op, val = item
            op = operator_map.get(op, op)
            var_name = f"V_{attr}_{idx}"
            if op == '=':
                body_preds.append(f'{attr}(X, "{val}")')
            elif op == '!=':
                body_preds.append(f'{attr}(X, {var_name}), {var_name} != "{val}"')
            elif op in ['<=', '>', '>=', '<']:
                # Scale numeric values
                if isinstance(val, (int, float)):
                    scaled_val = int(float(val) * scale_factor)
                    body_preds.append(f'{attr}(X, {var_name}), {var_name} {op} {scaled_val}')
                else:
                    body_preds.append(f'{attr}(X, "{val}")')
        for ab in ab_items:
            ab_head = ab[0]
            ab_val = ab[2]
            ab_pred = f'{ab_head}(X, "{ab_val}")'
            body_preds.append(f'not {ab_pred}')
        rule_str = f'{head_pred} :- {", ".join(body_preds)}.'
        asp_rules.append(rule_str)
    return '\n'.join(asp_rules)

def data_to_asp_facts(data, model, scale_factor=10):
    facts = []
    for idx, x in enumerate(data):
        patient_id = f'patient{idx}'
        x['id'] = patient_id
        facts.append(f'patient("{patient_id}").')
        for attr in model.attrs:
            if attr in x:
                val = x[attr]
                if isinstance(val, str):
                    val_escaped = val.replace('"', '\\"')
                    facts.append(f'{attr}("{patient_id}", "{val_escaped}").')
                else:
                    # Scale numeric values
                    scaled_val = int(float(val) * scale_factor)
                    facts.append(f'{attr}("{patient_id}", {scaled_val}).')
    return '\n'.join(facts)

def run_clingo(rules_str, facts_str, model):
    ctl = clingo.Control()
    asp_program = rules_str + "\n" + facts_str
    ctl.add("base", [], asp_program)
    ctl.ground([("base", [])])

    labels = []
    def on_model(m):
        symbols = m.symbols(shown=True)
        for sym in symbols:
            if sym.name == model.label:
                patient_id = sym.arguments[0].string
                label_value = sym.arguments[1].string
                labels.append((patient_id, label_value))
    
    ctl.configuration.solve.models = "0"
    ctl.solve(on_model=on_model)
    return labels

def predict_foldrpp_clingo(model, data_test, scale_factor=10):
    rules_str = model_to_asp(model, scale_factor)
    facts_str = data_to_asp_facts(data_test, model, scale_factor)
    asp_program = rules_str + "\n" + facts_str

    labels = run_clingo(rules_str, facts_str, model)
    label_map = {x['id']: x['label'] for x in data_test}
    y_pred = [0] * len(data_test)
    patient_id_to_index = {x['id']: idx for idx, x in enumerate(data_test)}
    for patient_id, label_value in labels:
        idx = patient_id_to_index[patient_id]
        y_pred[idx] = 1 if label_value == model.pos_val else 0
    return y_pred, asp_program  # Return the ASP program as a string
