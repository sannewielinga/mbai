from foldrpp import Foldrpp


def autism():
    """
    Autism dataset from UCI Machine Learning Repository.

    This dataset is a classification task. The label is the diagnosis of autism ('YES' or 'NO'). The features are:

        * a1 to a10: 10 items from the Autism Quotient (AQ) test,
            where 1 indicates 'definitely agree' and 0 indicates 'definitely disagree'.
        * age: the age of the individual.
        * gender: the gender of the individual ('f' or 'm').
        * ethnicity: the ethnicity of the individual ('white' or 'non-white').
        * jaundice: whether the individual suffered from jaundice ('yes' or 'no').
        * pdd: whether the individual was diagnosed with PDD-NOS ('yes' or 'no').
        * used_app_before: whether the individual had used the app before ('yes' or 'no').
        * relation: the relation of the individual to the person filling out the application ('Parent', 'Self', 'Relative', or 'Other').

    The dataset contains 704 instances.

    :return: A Foldrpp object and a list of dictionaries representing the dataset.
    """
    str_attrs = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "a9",
        "a10",
        "gender",
        "ethnicity",
        "jaundice",
        "pdd",
        "used_app_before",
        "relation",
    ]
    num_attrs = ["age"]
    label, pos_val = "label", "NO"
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data("data/autism/autism.csv")
    print("\n% autism dataset", len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def breastw():
    """
    The Wisconsin Breast Cancer Dataset (Diagnostic) is a dataset containing 569 instances each with 32 features.
    The features are:

    * clump_thickness: the clump thickness.
    * cell_size_uniformity: the uniformity of cell size.
    * cell_shape_uniformity: the uniformity of cell shape.
    * marginal_adhesion: the marginal adhesion.
    * single_epi_cell_size: the size of single epithelial cell.
    * bare_nuclei: the bare nuclei.
    * bland_chromatin: the bland chromatin.
    * normal_nucleoli: the normal nucleoli.
    * mitoses: the mitoses.

    The label is 'malignant' or 'benign'.
    The dataset contains 569 instances.

    :return: A Foldrpp object and a list of dictionaries representing the dataset.
    """
    str_attrs = [
        "clump_thickness",
        "cell_size_uniformity",
        "cell_shape_uniformity",
        "marginal_adhesion",
        "single_epi_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
    ]
    num_attrs = []
    label, pos_val = "label", "malignant"
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data("data/breastw/breastw.csv")
    print("\n% breastw dataset", len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def heart():
    """
    Heart Disease dataset from UCI Machine Learning Repository.

    This dataset is a classification task. The label indicates the presence of heart disease ('absent' or 'present'). The features include:

        * sex: the sex of the individual.
        * chest_pain: the type of chest pain experienced.
        * fasting_blood_sugar: whether fasting blood sugar is greater than 120 mg/dl (1 = true, 0 = false).
        * resting_electrocardiographic_results: results of the resting electrocardiogram.
        * exercise_induced_angina: whether the individual experiences angina during exercise (1 = yes, 0 = no).
        * slope: the slope of the peak exercise ST segment.
        * major_vessels: number of major vessels colored by fluoroscopy.
        * thal: a blood disorder called thalassemia.
        * age: the age of the individual.
        * blood_pressure: resting blood pressure in mm Hg.
        * serum_cholestoral: serum cholesterol in mg/dl.
        * maximum_heart_rate_achieved: maximum heart rate achieved.
        * oldpeak: ST depression induced by exercise relative to rest.

    The dataset contains data instances with these attributes.

    :return: A Foldrpp object and a list of dictionaries representing the dataset.
    """
    str_attrs = [
        "sex",
        "chest_pain",
        "fasting_blood_sugar",
        "resting_electrocardiographic_results",
        "exercise_induced_angina",
        "slope",
        "major_vessels",
        "thal",
    ]
    num_attrs = [
        "age",
        "blood_pressure",
        "serum_cholestoral",
        "maximum_heart_rate_achieved",
        "oldpeak",
    ]
    label, pos_val = "label", "absent"
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data("data/heart/heart.csv")
    print("\n% heart dataset", len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def kidney():
    """
    The kidney dataset is a classification dataset of 400 instances.
    Each instance consists of the following attributes:

    * al: albumin.
    * su: sugar.
    * rbc: red blood cells.
    * pc: pus cell.
    * pcc: pus cell clumps.
    * ba: bacteria.
    * htn: hypertension.
    * dm: diabetes mellitus.
    * cad: coronary artery disease.
    * appet: appetite.
    * pe: pedal edema.
    * ane: anemia.
    * age: age in years.
    * bp: blood pressure.
    * sg: specific gravity.
    * bgr: blood glucose random.
    * bu: blood urea.
    * sc: serum creatinine.
    * sod: sodium.
    * pot: potassium.
    * hemo: hemoglobin.
    * pcv: packed cell volume.
    * wbcc: white blood cell count.
    * rbcc: red blood cell count.

    The dataset contains data instances with these attributes.

    :return: A Foldrpp object and a list of dictionaries representing the dataset.
    """
    str_attrs = [
        "al",
        "su",
        "rbc",
        "pc",
        "pcc",
        "ba",
        "htn",
        "dm",
        "cad",
        "appet",
        "pe",
        "ane",
    ]
    num_attrs = [
        "age",
        "bp",
        "sg",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc",
    ]
    label, pos_val = "label", "ckd"
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data("data/kidney/kidney.csv")
    print("\n% kidney dataset", len(data), len(str_attrs + num_attrs) + 1)
    return model, data


def ecoli():
    """
    The Ecoli dataset contains 336 instances each with 7 attributes representing
    the properties of proteins in the E. coli bacterium. The label is 'cp' or 'im'.
    The attributes are:

    * sn: the sequence name.
    * mcg: the McGeoch method for signal sequence recognition.
    * gvh: the von Heijne method for signal sequence recognition.
    * lip: the decision whether the protein is a lipoprotein or not.
    * chg: the decision whether the protein is a charged protein or not.
    * aac: the amino acid composition.
    * alm1: the score produced by the ALOM membrane protein topology and signal
        peptide prediction method.
    * alm2: the score produced by the PHDacc method.

    The dataset contains 336 instances.

    :return: A Foldrpp object and a list of dictionaries representing the dataset.
    """
    str_attrs = ["sn"]
    num_attrs = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
    label, pos_val = "label", "cp"
    model = Foldrpp(str_attrs, num_attrs, label, pos_val)
    data = model.load_data("data/ecoli/ecoli.csv")
    print("\n% ecoli dataset", len(data), len(str_attrs + num_attrs) + 1)
    return model, data
