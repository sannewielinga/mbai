from foldrpp import Foldrpp

def adult():
    str_attrs = ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
                 'race', 'sex', 'native_country']
    num_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    label, pos_val = 'label', '<=50K'
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data('data/adult/adult.csv')
    print('\n% adult dataset', len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def autism():
    str_attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'gender', 'ethnicity', 'jaundice',
                 'pdd', 'used_app_before', 'relation']
    num_attrs = ['age']
    label, pos_val = 'label', 'NO'
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data('data/autism/autism.csv')
    print('\n% autism dataset', len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def breastw():
    str_attrs = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
                 'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    num_attrs = []
    label, pos_val = 'label', 'malignant'
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data('data/breastw/breastw.csv')
    print('\n% breastw dataset', len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def heart():
    str_attrs = ['sex', 'chest_pain', 'fasting_blood_sugar',
                 'resting_electrocardiographic_results', 'exercise_induced_angina',
                 'slope', 'major_vessels', 'thal']
    num_attrs = ['age', 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak']
    label, pos_val = 'label', 'absent'
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data('data/heart/heart.csv')
    print('\n% heart dataset', len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def kidney():
    str_attrs = ['al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    num_attrs = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    label, pos_val = 'label', 'ckd'
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data('data/kidney/kidney.csv')
    print('\n% kidney dataset', len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def ecoli():
    str_attrs = ['sn']
    num_attrs = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']
    label, pos_val = 'label', 'cp'
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data('data/ecoli/ecoli.csv')
    print('\n% ecoli dataset', len(data), len(str_attrs + num_attrs) + 1)
    return model, data
