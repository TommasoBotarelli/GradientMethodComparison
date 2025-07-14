import pycutest
from GradientMethod import *
from StepSizeMethods import *
import json
import time
import os


MAX_ITERATIONS = 10
SKIP_NEW_CALCULATION_AFTER_S = 30

def get_test_problems(problem_file):
    problems = []
    
    with open(problem_file) as f:
        for x in f:
            problems.append(x)

    return problems


def read_json_to_dict(file_path):
    data = {}

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    return data


def save_results(file_path, result, problem_name):
    data = read_json_to_dict(file_path)
    data[problem_name] = {}
    data[problem_name] = result

    with open(file_path, "w") as jsonfile:
        json.dump(data, jsonfile, indent=2)



if __name__ == "__main__":
    all_problems = pycutest.find_problems(objective='other', constraints='unconstrained', regular=True)
    result_file_name = "results"
    problem_file = "problems.txt"
    selected_problems = get_test_problems(problem_file)
    
    constant_step_method = ConstantStep(1)
    barzilai_borwein_1 = BarzilaiBorwein(1)
    barzilai_borwein_2 = BarzilaiBorwein(2)
    no_particular_step_size = NoStepSize()

    max_steps = 5000
    tolerance = 1e-3

    all_methods = [Lbfgs(step_size_method=no_particular_step_size, tolerance=tolerance, max_steps=max_steps),
                   GradientMethodArmijo(step_size_method=constant_step_method, tolerance=tolerance, max_steps=max_steps),
                   GradientMethodGrippo(step_size_method=constant_step_method, tolerance=tolerance, max_steps=max_steps, M=10), 
                   GradientMethodGrippo(step_size_method=constant_step_method, tolerance=tolerance, max_steps=max_steps, M=5), 
                   GradientMethodNLSA(step_size_method=constant_step_method, tolerance=tolerance, max_steps=max_steps), 
                   GradientMethodArmijo(step_size_method=barzilai_borwein_1, tolerance=tolerance, max_steps=max_steps), 
                   GradientMethodGrippo(step_size_method=barzilai_borwein_1, tolerance=tolerance, max_steps=max_steps, M=10),
                   GradientMethodGrippo(step_size_method=barzilai_borwein_1, tolerance=tolerance, max_steps=max_steps, M=5),
                   GradientMethodNLSA(step_size_method=barzilai_borwein_1, tolerance=tolerance, max_steps=max_steps),
                   GradientMethodArmijo(step_size_method=barzilai_borwein_2, tolerance=tolerance, max_steps=max_steps), 
                   GradientMethodGrippo(step_size_method=barzilai_borwein_2, tolerance=tolerance, max_steps=max_steps, M=10),
                   GradientMethodGrippo(step_size_method=barzilai_borwein_2, tolerance=tolerance, max_steps=max_steps, M=5),
                   GradientMethodNLSA(step_size_method=barzilai_borwein_2, tolerance=tolerance, max_steps=max_steps)]

    for problem in selected_problems:
        moving_problem_result = {}
        
        for iterations in range(MAX_ITERATIONS):
            for method in all_methods:
                method_name = method.get_name()
                step_size_method_name = method.get_step_method().get_name()

                if method_name not in list(moving_problem_result.keys()):
                    moving_problem_result[method_name] = {}
                if step_size_method_name not in list(moving_problem_result[method_name].keys()):
                    moving_problem_result[method_name][step_size_method_name] = []
                
                if len(moving_problem_result[method_name][step_size_method_name]) == 0 or moving_problem_result[method_name][step_size_method_name][-1]["execution_time"] < SKIP_NEW_CALCULATION_AFTER_S:
                    method.reset()
                    pycutest.clear_cache(problem)
                    clean_problem = pycutest.import_problem(problem)

                    start = time.time()
                    method.optimize(clean_problem)
                    end = time.time()
                    print(f"Execution time for problem {clean_problem.name} with {method_name} and {step_size_method_name} = {end-start}")
                    print("")
                    
                    single_result = {}

                    single_result["iterations"] = method.get_iterations()
                    single_result["value"] = method.get_final_value()
                    single_result["execution_time"] = end - start

                    moving_problem_result[method_name][step_size_method_name].append(single_result)
                else:
                    print(f"SKIP {problem.replace("\n", "")} with {method_name} and {step_size_method_name}")
                    print("")
            
        save_results(f"{result_file_name}.json", moving_problem_result, problem)
        
