from helpers.logging_config import logger

import json


from agents.unittest_agent.unittest_agent import UnitTestAgent
from helpers.analysers.method_formatter_analyzer import MethodFormatterAnalyzer
from helpers.check_syntax import check_syntax
from helpers.clean_code import clean_code
from helpers.read_file import read_file
from helpers.save_code import save_code


MAX_AGENT_FLOW_COUNT = 3


def workflow(llm_instance, file):
    try:
        if check_syntax(read_file(file)):
            code_analyzer = MethodFormatterAnalyzer()

            analysis_data = code_analyzer.format_analysis(file)
            # logger.debug(
            #     "Analysis Data:\n" + json.dumps(analysis_data, indent=4)
            # )  

            unit_test_agent = UnitTestAgent(llm_instance)
            for methods in analysis_data:
                code =  unit_test_agent.invoke(methods)
                i=0
                while (not check_syntax(code) and i<MAX_AGENT_FLOW_COUNT):
                    code = unit_test_agent.invoke(methods)
                    i+=1

                if not check_syntax(code):
                    return "not able to genrate unittest"

                save_code(
                    "./generated_code/unittest.py", code
                )
            return "unittest is generated"

        else:
            return "your code is not valid"
    except Exception as e:
        print(e)
