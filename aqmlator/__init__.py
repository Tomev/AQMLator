"""
    Machine learning (ML) is one of the fastest-progressing research directions in
    applied computer science. The field investigates the development of algorithms that
    can learn from data by fitting a collection of model parameters to the data by
    iteratively optimizing an objective function. The selection of a model structure, be
    it a neural network topology or kernel function, is a problem-dependent task often
    made by hand. But there exist Auto ML systems that can choose a model automatically
    depending solely on the input data and task.

    Quantum computing (QC) studies how hard computational problems can be efficiently
    solved using quantum mechanics (QM). A large-scale error-corrected quantum computer
    can solve computational problems that don't have a classical solution. A prime
    example of that is Shor's algorithm for integer factorization. The
    'holy grail' of applied QC is the so-called quantum supremacy or quantum
    advantage. That is the name for a technological milestone that would mark the moment
    when quantum machines will solve a specific task faster than the most advanced
    supercomputer. Although there have already been several quantum supremacy claims in
    recent years, there are no practical problems solved using quantum computing yet.

    The search for such practical problems focuses on applications in the soft computing
    areas that are less susceptible to current quantum hardware imperfections. One of
    the possible applications of QC is quantum machine learning (QML). Quantum computers
    can be employed to build ML models that can be fit to data and then used during the
    inference process. In one of the QML scenarios, a variational quantum circuit
    forming a quantum neural network (QNN) constitutes only one part of the ML data
    processing pipeline. Since designing such a pipeline with a quantum component is
    challenging for non-experts in QC, we propose an auto ML solution that uses QML
    techniques.

    We call our solution (A)uto (Q)uantum (M)achine (L)earning platform---AQMLator. It
    aims to enable data scientists, especially those without knowledge of QC, to use
    QML models. Given the data, the system will seek the model that fits the
    data best. Additionally, it will propose the circuit structure of the
    VQC model and its weights. Ultimately, the user can use the model as-is
    or encompass it in a hybrid model e.g. by extracting the proposed VQC model as a
    torch layer.
"""
