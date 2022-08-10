import numpy as np
import pandas as pd
import tensorflow as tf
import sys

GREEN = '\033[32m'
RED = '\033[31m'
BLACK = '\033[30m'

###########################################################################

# PROJECT 0:

def test_square_function(func):
    x_inputs = np.random.uniform(1,100, 10)
    expected_outputs = x_inputs**2
    if (func(x_inputs) == expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: x = {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                x_inputs, expected_outputs, func(x_inputs)))

def test_normal_pdf(func):
    inputs = ((-2, 0.1, -1.5), (-1, 0.2, -1.4), (2, 0.5, 3.5))
    expected_outputs = [1.4867195147343004e-05, 0.2699548325659406, 0.008863696823876015]
    outputs = [func(*x) for x in inputs]
    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                inputs, expected_outputs, outputs))

def test_is_even(func):
    x_inputs = np.random.randint(-100, 100, 10)
    expected_outputs = x_inputs % 2 == 0
    outputs = [func(x) for x in x_inputs]
    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                x_inputs, expected_outputs, outputs))

def test_sum_of_squares(func):
    x_inputs = np.random.uniform(-10, 10, size = (10, 5))
    expected_outputs = np.sum(x_inputs**2, axis = 1)
    outputs = [func(x) for x in x_inputs]
    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                x_inputs, expected_outputs, outputs))

def test_taylor_exp(func):
    inputs = ((3, 50), (-1, 20), (0.5, 30))
    expected_outputs = [20.08553692318766, 0.36787944117144245, 1.6487212707001278]
    outputs = [func(*x) for x in inputs]
    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                inputs, expected_outputs, outputs))

def test_is_prime(func):
    n_inputs = np.arange(2,50)
    expected_outputs = np.array([ True,  True, False,  True, False,  True, False, False, False,
        True, False,  True, False, False, False,  True, False,  True,
       False, False, False,  True, False, False, False, False, False,
        True, False,  True, False, False, False, False, False,  True,
       False, False, False,  True, False,  True, False, False, False,
        True, False, False])
    outputs = np.array([func(n) for n in n_inputs])
    if (outputs == expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                n_inputs, expected_outputs, outputs))

def test_factorize(func):
    n_inputs = np.arange(2,100)
    expected_outputs = np.array([list([2]), list([3]), list([2, 2]), list([5]), list([2, 3]),
                                list([7]), list([2, 2, 2]), list([3, 3]), list([2, 5]), list([11]),
                                list([2, 2, 3]), list([13]), list([2, 7]), list([3, 5]),
                                list([2, 2, 2, 2]), list([17]), list([2, 3, 3]), list([19]),
                                list([2, 2, 5]), list([3, 7]), list([2, 11]), list([23]),
                                list([2, 2, 2, 3]), list([5, 5]), list([2, 13]), list([3, 3, 3]),
                                list([2, 2, 7]), list([29]), list([2, 3, 5]), list([31]),
                                list([2, 2, 2, 2, 2]), list([3, 11]), list([2, 17]), list([5, 7]),
                                list([2, 2, 3, 3]), list([37]), list([2, 19]), list([3, 13]),
                                list([2, 2, 2, 5]), list([41]), list([2, 3, 7]), list([43]),
                                list([2, 2, 11]), list([3, 3, 5]), list([2, 23]), list([47]),
                                list([2, 2, 2, 2, 3]), list([7, 7]), list([2, 5, 5]),
                                list([3, 17]), list([2, 2, 13]), list([53]), list([2, 3, 3, 3]),
                                list([5, 11]), list([2, 2, 2, 7]), list([3, 19]), list([2, 29]),
                                list([59]), list([2, 2, 3, 5]), list([61]), list([2, 31]),
                                list([3, 3, 7]), list([2, 2, 2, 2, 2, 2]), list([5, 13]),
                                list([2, 3, 11]), list([67]), list([2, 2, 17]), list([3, 23]),
                                list([2, 5, 7]), list([71]), list([2, 2, 2, 3, 3]), list([73]),
                                list([2, 37]), list([3, 5, 5]), list([2, 2, 19]), list([7, 11]),
                                list([2, 3, 13]), list([79]), list([2, 2, 2, 2, 5]),
                                list([3, 3, 3, 3]), list([2, 41]), list([83]), list([2, 2, 3, 7]),
                                list([5, 17]), list([2, 43]), list([3, 29]), list([2, 2, 2, 11]),
                                list([89]), list([2, 3, 3, 5]), list([7, 13]), list([2, 2, 23]),
                                list([3, 31]), list([2, 47]), list([5, 19]),
                                list([2, 2, 2, 2, 2, 3]), list([97]), list([2, 7, 7]),
                                list([3, 3, 11])], dtype=object)
    outputs = np.array([func(n) for n in n_inputs])
    check = True
    for i in range(len(n_inputs)):
        if expected_outputs[i] != sorted(outputs[i]):
            check = False
            if not check:
                print(RED + 'Test did not pass' + BLACK)
                print('For the input: {} \nThe expected output is {} \nYour function output is {}'.format(
                        n_inputs[i], expected_outputs[i], outputs[i]))
    if check:
        print(GREEN + 'Test passed')

###########################################################################

# PROJECT 1:

def test_column_means(func):

    inputs = np.random.uniform(-20,20, size = (10,6))

    expected_output = np.mean(inputs, axis = 0)

    output = func(inputs)

    assert expected_output.shape == output.shape, "For the input {} the expected \
output shape is {} but got output of shape {}".format(inputs, expected_output.shape, output.shape)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input: {} \nThe expected outpu is {} \nYour function output is {}'.format(
                inputs, expected_output, output))

def test_cutoff(func):

    x_in = [(np.array([[0.28897937, 0.43569014, 0.36364435, 0.39218248],
             [0.66244581, 0.37877839, 0.3410136,  0.01200013],
             [0.62660962, 0.88664802, 0.39039634, 0.29926516]]), 0.5),
           (np.array([[0.23448093, 0.05224714],
                   [0.20100718, 0.86350585],
                   [0.55008888, 0.4784401 ],
                   [0.97934747, 0.15784187],
                   [0.55201495, 0.47493771]]), 0.6)
           ]


    expected_outputs = [np.array([[0.28897937, 0.43569014, 0.36364435, 0.39218248],
                                   [0.5       , 0.37877839, 0.3410136 , 0.01200013],
                                   [0.5       , 0.5       , 0.39039634, 0.29926516]]),
                        np.array([[0.23448093, 0.05224714],
                                   [0.20100718, 0.6       ],
                                   [0.55008888, 0.4784401 ],
                                   [0.6       , 0.15784187],
                                   [0.55201495, 0.47493771]])
                       ]

    check = True
    for i, (x, th) in enumerate(x_in):
        output = func(x, th)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For the input array {} and threshold {}, the expected output is {}, your output is {}'.format(x, th, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed')

def test_max_index(func):

    x_in = [np.array([[0.59924692, 0.28694015, 0.16983537, 0.89984641],
                       [0.03111706, 0.42997205, 0.54268424, 0.64534421],
                       [0.16471448, 0.75338066, 0.54912713, 0.70684244]]),
            np.array([[0.03367516, 0.45768455, 0.46902002, 0.24923532],
                       [0.21920534, 0.14771333, 0.24763095, 0.35050109],
                       [0.11549889, 0.65836753, 0.53742414, 0.8707693 ],
                       [0.8828907 , 0.06985487, 0.21786648, 0.651936  ],
                       [0.09754203, 0.58064407, 0.82283824, 0.08302386],
                       [0.70813905, 0.37341553, 0.0143709 , 0.85784191]])
           ]

    expected_outputs = [np.array([3, 3, 1]),
                       np.array([2, 3, 3, 0, 2, 3])]

    check = True
    for i, x in enumerate(x_in):
        output = func(x)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For the input array {}, the expected output is {}, your output is {}'.format(x, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed')

def test_linear(func):

    x_in = [(np.array([[0.46527354, 0.31002314, 0.0407845 ],
                       [0.45594427, 0.67337049, 0.51686615],
                       [0.02210991, 0.98308284, 0.5778303 ]]),
             np.array([[0.12307017, 0.79964265, 0.60164791]]),
             np.array([[0.41128018, 0.58183524, 0.89999135]])),
            (np.array([[0.88527927, 0.71755922, 0.49300952, 0.90611223],
                       [0.15506991, 0.91299081, 0.11205118, 0.85878849],
                       [0.78393805, 0.30408309, 0.21298612, 0.20433642]]),
             np.array([[0.53660877, 0.57383126, 0.10156138]]),
             np.array([[0.11599683, 0.22034654, 0.66471019, 0.02979967]]))
           ]


    expected_outputs = [np.array([[0.84643634, 1.74991534, 1.66596931]]),
                        np.array([[0.75964724, 1.16018088, 1.01519306, 1.02957981]])]



    check = True
    for i, (W,x,b) in enumerate(x_in):
        output = func(W,x,b)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For W={}, x={} and b={}, the expected output is {}, your output is {}'.format(W, x, b, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed')

def test_uniform_matrix(func):

    x_in = [(2,3),
            (1,3),
            (2,2)]


    expected_outputs = [np.array([[-0.16595599,  0.44064899, -0.99977125],
                                  [-0.39533485, -0.70648822, -0.81532281]]),
                        np.array([[-0.16595599,  0.44064899, -0.99977125]]),
                        np.array([[-0.16595599,  0.44064899],
                                  [-0.99977125, -0.39533485]])]

    check = True
    for i, x in enumerate(x_in):
        output = func(*x)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For x={}, the expected output is {}, your output is {}'.format(x, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed')

def test_orthogonal(func):

    v1 = [np.array([1,0,0]), np.array([1,2,3])]
    v2 = [np.array([0, 0.5, 0.7]), np.array([-2,5,6])]

    expected_outputs = np.array([True, False])

    check = True
    for i, (u, v) in enumerate(zip(v1, v2)):
        output = func(u, v)
        if expected_outputs[i] != output:
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For the inputs {} and {}, the expected output is {}, your output is {}'.format(u, v, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed')

def test_rotation(func):

    inputs_theta = [0.5, 1.0, 5.0]

    expected_outputs = [np.array([[ 0.87758256, -0.47942554],
                                  [ 0.47942554,  0.87758256]]),
                        np.array([[ 0.54030231, -0.84147098],
                                  [ 0.84147098,  0.54030231]]),
                        np.array([[ 0.28366219,  0.95892427],
                                  [-0.95892427,  0.28366219]])
                        ]


    check = True
    for i, theta in enumerate(inputs_theta):
        output = func(theta)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For the input {}, the expected output is {}, your output is {}'.format(theta, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed')

def test_prediction(func):
    inputs_w = np.array([[-0.45176045,  0.12105998],
                        [ 0.34345957, -0.29514074],
                        [ 0.71165673, -0.60992503],
                        [ 0.49464162, -0.42079451],
                        [ 0.54759857, -0.14452534],
                        [ 0.61539682, -0.29293027],
                        [-0.57261352,  0.53456902],
                        [-0.3827161 ,  0.46649014],
                        [ 0.48894631, -0.5572066 ],
                        [-0.57177573, -0.60210415]])
    inputs_x = np.array([[-0.71496332, -0.2458348 ],
                        [-0.94674423, -0.77815926],
                        [ 0.34912805,  0.59955307],
                        [-0.83894095, -0.53659538],
                        [-0.58474868,  0.83466713],
                        [ 0.42262904,  0.10776922],
                        [-0.39096402,  0.6697081 ],
                        [-0.12938808,  0.84691243],
                        [ 0.41210361, -0.04393738],
                        [-0.74757979,  0.9520871 ]])

    expected_outputs = [1, -1, -1, -1, -1, 1, 1, 1, 1, -1]

    output = [func(w, x) for w, x in zip(inputs_w, inputs_x)]

    check = True
    for i,(w,x) in enumerate(zip(inputs_w, inputs_x)):
        output = func(w,x)
        if output != expected_outputs[i]:
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For the inputs w={} and x={} \nThe expected output is {} \nYour function output is {}'.format(w, x, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed')

def test_perceptron(func):
    x = np.array([[-1,1], [0,1], [1,0], [2,1]])
    y = np.array([-1, -1, 1, 1])

    expected_output_version_1 = np.array([ 1, 0])
    expected_output_version_2 = np.array([1, -1])

    output = func(x,y)

    if np.isclose(expected_output_version_1, output).all() or np.isclose(expected_output_version_2, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: x={}, y={} \nThe expected output is {} or {} depending on your update procedure, \nYour function output is {}'.format(
                x, y, expected_output_version_1, expected_output_version_2, output))

###########################################################################

# PROJECT 2:

def test_preprocess_sk(func):
    x_inputs, y_inputs = np.random.uniform(1,100, (10,6)), np.random.choice([-1,1],(10,))
    expected_outputs = np.append(x_inputs, np.ones((x_inputs.shape[0],1)), axis=1)
    expected_outputs = expected_outputs * y_inputs.reshape(-1,1)
    if (func(x_inputs, y_inputs) == expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print(f'For the inputs: x = {x_inputs} and y = {y_inputs}\nThe expected outputs are {expected_outputs} \nYour function outputs are {func(x_inputs, y_inputs)}')

def test_forward_pass_linear_regression(func):
    np.random.seed(seed = 1)

    inputs = np.random.randn(4,5)

    np.random.seed(seed = 1)
    weights = np.random.uniform(low = -1, high = 1, size = (5, 1))
    biases = np.zeros((1, 1))

    expected_output = np.array([[-0.19830715],
                               [ 1.9618864 ],
                               [-1.47726572],
                               [ 0.55577019]])

    output = func(inputs, weights, biases)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input = {}, \n weights = {} and, \n biases = {}, the expected output is {}, your output is {}'.format(inputs, weights, biases, expected_output, output))

def test_coin_toss(func):

    np.random.seed(seed = 1)
    inputs = (100, 0.9)

    expected_output = np.random.binomial(1, 0.9, size = 100)

    output = func(*inputs)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input: {} \nThe expected output {} \nYour function output is {}'.format(
                inputs, expected_output, output))

def test_die_roll(func):

    np.random.seed(seed = 1)
    m, p = 30, [1/6]*6

    expected_output = np.random.multinomial(1, p, size = m)
    output = func(m, p)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                m, p, expected_output, output))

def test_expected_value(func):

    inputs_f = np.array([2,3,4,5,6,7,8,9,10,11,12])
    inputs_P = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

    expected_output = 7.

    output = func(inputs_f, inputs_P)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                inputs_f, inputs_P, expected_output, output))

def test_kl_divergence(func):

    inputs_P, inputs_Q = np.array([0.1, 0.2, 0.7]), np.array([0.7, 0.21, 0.09])

    expected_output = 1.23154041755978

    output = func(inputs_P, inputs_Q)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                inputs_P, inputs_Q, expected_output, output))

###########################################################################

# PROJECT 3:

###########################################################################

# PROJECT 4:

###########################################################################

# PROJECT 5:

def test_reduce_k_dim(func):
    x_inputs, V_input = np.random.choice(range(100), (6,3)), np.random.choice(range(100), (3,3))
    expected_outputs = [np.matmul(x_inputs, V_input[:,:k]) for k in range(1,4)]
    try:
        if min([(func(x_inputs, V_input, k) == expected_outputs[k-1]).all() for k in range(1,4)]) == 1:
            print(GREEN + 'Test passed')
    except:
        print(RED + 'Test did not pass' + BLACK)
        print(f'For the inputs: x_train = {x_inputs}, and V = {V_input}\nThe expected outputs are {expected_outputs} \nYour function outputs are {[func(x_inputs, V_input, k) for k in range(1,4)]}')

###########################################################################

# Uneeded as of now:

def test_derivative(func):

    inputs_f = lambda x: x**2
    inputs_x = 1

    expected_output = 2

    output = func(inputs_f, inputs_x)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                'f(x)=x^2', inputs_x, expected_output, output))

def test_gradient_descent(func):

    f = lambda x: x**2

    g = lambda x: np.cos(x)

    functions = [f, g]

    names = ['x^2', 'cos(x)']
    N = 1000
    learning_rate = 0.01

    expected_outputs = [[-4.999298157164008e-06, 2.4992982064223448e-11], [3.141173062193505, -0.9999999119715314]]

    check = True

    for i in range(2):
        output = func(functions[i], learning_rate, N)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For the function {}, learning rate 0.01 and number of iterations 1000; the expected output is {}, your output is {}'.format(names[i], expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed' + BLACK)

def test_derivative_tf(func):

    inputs_f = lambda x: x**2
    inputs_a = 1.

    expected_output = 2.

    output = func(inputs_f, inputs_a).numpy()

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                'f(x)=x^2', inputs_a, expected_output, output))

def test_gradient_descent_tf(func):

    f = lambda x: x**2

    g = lambda x: tf.cos(x)

    functions = [f, g]

    names = ['x^2', 'cos(x)']
    N = 1000
    learning_rate = 0.01

    expected_outputs = [[1.1184361e-09, 1.2508993e-18], [ 3.1413395,  -0.99999994]]

    check = True

    for i in range(2):
        output_x, output_f = func(functions[i], learning_rate, N)
        output = np.c_[output_x.numpy(), output_f.numpy()]
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For the function {}, learning rate 0.01 and number of iterations 1000; the expected output is {}, your output is {}'.format(names[i], expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed' + BLACK)

def test_initialize_weights_and_biases(func):

    inputs = [12, 2]

    expected_output_W, expected_output_b = np.array([[-0.1086437 ,  0.28847248],
                                [-0.65450392, -0.25880741],
                                [-0.46250511, -0.53375407],
                                [-0.41078181, -0.20220847],
                                [-0.1351631 ,  0.05082303],
                                [-0.10579922,  0.24250925],
                                [-0.38696284,  0.49507194],
                                [-0.61879489,  0.22319436],
                                [-0.10827343,  0.07684302],
                                [-0.47084402, -0.39527794],
                                [ 0.39376707,  0.61309832],
                                [-0.24428509,  0.25180941]]), np.array([[0., 0.]])


    output_W, output_b = func(*inputs)

    if np.isclose(expected_output_W, output_W).all() and np.isclose(expected_output_b, output_b).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the inputs: {} \nThe expected output {}, {} \nYour function output is {}, {}'.format(
                inputs, expected_output_W, expected_output_b, output_W, output_b))

def test_activations(func):

    x_in = [np.array([0.30233257263183977]),
            np.array([0.18626021, 0.34556073]),
            np.array([-0.10961306, -0.05189851,  0.55074457,  0.71826158,  0.06342418])
           ]


    expected_outputs = {'sigmoid': [np.array([0.57501263]),
                                    np.array([0.5464309 , 0.58554065]),
                                    np.array([0.47262414, 0.48702828, 0.63430832, 0.67222409, 0.51585073])],
                        'sigmoid_derivative' : [np.array([0.2443731]),
                                                np.array([0.24784417, 0.2426828 ]),
                                                np.array([0.24925056, 0.24983173, 0.23196128, 0.22033886, 0.24974875])],
                        'relu': [np.array([0.30233257]),
                                 np.array([0.18626021, 0.34556073]),
                                 np.array([0.        , 0.        , 0.55074457, 0.71826158, 0.06342418])],
                        'relu_derivative': [np.array([1.]),
                                            np.array([1., 1.]),
                                            np.array([0., 0., 1., 1., 1.])],
                        'softmax': [np.array([1.]),
                                    np.array([0.46025888, 0.53974112]),
                                    np.array([0.13382837, 0.14177946, 0.25902273, 0.30625951, 0.15910994])]
                       }
    check = True
    for i, x in enumerate(x_in):
        output = func(x)
        if not np.isclose(expected_outputs[func.__name__][i], output).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For x={}, the expected output is {}, your output is {}'.format(x, expected_outputs[func.__name__][i], output))
    if check:
        print(GREEN + 'Test passed' + BLACK)

def test_loss(func):

    np.random.seed(seed = 1)
    y = np.random.randint(0,2,50).reshape(-1,1)
    y_hat = np.random.random(50).reshape(-1,1)

    expected_output = 0.8862180846119079

    output = func(y_hat, y)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For y_hat = {} and y = {}, the expected output is {}, your output is {}'.format(y_hat, y, expected_output, output))

def test_forward_pass(func):
    np.random.seed(seed = 1)

    inputs = np.random.randn(4,5)

    np.random.seed(seed = 1)
    weights = np.random.uniform(low = -1, high = 1, size = (5, 1))
    biases = np.zeros((1, 1))

    expected_output = np.array([[0.45058505],
                             [0.87673696],
                             [0.18584077],
                             [0.63547328]])

    output = func(inputs, weights, biases)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input = {}, \n weights = {} and, \n biases = {}, the expected output is {}, your output is {}'.format(inputs, weights, biases, expected_output, output))

def test_crossentropy_loss(func):
    np.random.seed(seed = 1)
    y = np.random.randint(0,2,4)
    y_hat = np.random.random(4)

    expected_outputs = np.array([-9.07602963, -1.19622763, -0.1587096 , -0.09688387])

    check = True
    for i in range(4):
        output = func(y_hat[i], y[i])
        if not np.isclose(expected_outputs[i], output):
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For y_hat={} and y = {}, the expected output is {}, your output is {}'.format(y_hat[i], y[i], expected_outputs[i], output))
    if check:
        print(GREEN + 'Test passed' + BLACK)

def test_update_parameters(func):

    np.random.seed(seed = 1)

    x = np.random.randn(4,5)
    y = np.random.randint(0,2,5)

    np.random.seed(seed = 1)
    weights = np.random.uniform(low = -1, high = 1, size = (5, 1))
    biases = np.zeros((1, 1))

    learning_rate = 0.01

    expected_outputs = [np.array([[-0.16167306],
                                    [ 0.43874483],
                                    [-1.00218754],
                                    [-0.39745354],
                                    [-0.70836915]]),
                        np.array([[-0.00137159]])]


    check = True
    outputs = func(x,y,weights,biases,learning_rate)
    for i in range(2):
        if not np.isclose(expected_outputs[i], outputs[i]).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For x={}, y = {}, weights = {}, biases = {} and learning_rate = {}, the expected outputs are {}, your outputs are {}'.format(x, y, weights, biases, learning_rate, expected_outputs, outputs))
            break
    if check:
        print(GREEN + 'Test passed' + BLACK)

def test_accuracy(func):
    input_y = np.array([[0],[0],[1],[1]])
    input_y_hat = np.array([[0.3],[0.7],[0.4],[0.8]])

    expected_output = 0.5

    output = func(input_y_hat, input_y)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For y_hat = {} and y = {}, the expected output is {}, your output is {}'.format(input_y_hat, input_y, expected_output, output))

def test_forward_pass_linear_regression(func):
    np.random.seed(seed = 1)

    inputs = np.random.randn(4,5)

    np.random.seed(seed = 1)
    weights = np.random.uniform(low = -1, high = 1, size = (5, 1))
    biases = np.zeros((1, 1))

    expected_output = np.array([[-0.19830715],
                               [ 1.9618864 ],
                               [-1.47726572],
                               [ 0.55577019]])

    output = func(inputs, weights, biases)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input = {}, \n weights = {} and, \n biases = {}, the expected output is {}, your output is {}'.format(inputs, weights, biases, expected_output, output))

def test_mse(func):
    np.random.seed(seed = 1)
    input_y_hat = np.random.randint(-10,10,10)
    np.random.seed(seed = 2)
    input_y = np.random.randint(-10,10,10)

    expected_output = 40.9

    output = func(input_y_hat, input_y)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For y_hat = {} and y = {}, the expected output is {}, your output is {}'.format(input_y_hat, input_y, expected_output, output))

def test_update_parameters_linear_reg(func):

    np.random.seed(seed = 1)
    x = np.random.randn(4,5)

    np.random.seed(seed = 1)
    y = np.random.randint(0,2,size=(4,1))

    np.random.seed(seed = 1)
    weights = np.random.uniform(low = -1, high = 1, size = (5, 1))
    biases = np.zeros((1, 1))

    learning_rate = 0.01

    expected_outputs = [np.array([[-0.13129853],
                                [ 0.41385435],
                                [-0.99921688],
                                [-0.40625204],
                                [-0.69334893]]),
                        np.array([[0.00578958]])]



    check = True
    outputs = func(x,y,weights,biases,learning_rate)
    for i in range(2):
        if not np.isclose(expected_outputs[i], outputs[i]).all():
            check = False
            print(RED + 'Test did not pass' + BLACK)
            print('For x={}, y = {}, weights = {}, biases = {} and learning_rate = {}, the expected outputs are {}, your outputs are {}'.format(x, y, weights, biases, learning_rate, expected_outputs, outputs))
            break
    if check:
        print(GREEN + 'Test passed' + BLACK)

def test_explicit_solution(func):
    np.random.seed(seed = 2)
    x_train = np.random.random([100])
    y_train = 5*x_train - 3 + np.random.randn(100)

    x_train = x_train.reshape(-1,1)
    y_train = y_train.reshape(-1,1)


    expected_output = np.array([[ 5.795226  ],
                                [-3.25893883]])
    output = func(x_train, y_train)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed' + BLACK)
    else:
        print(RED + 'Test did not pass' + BLACK)

###########################################################################