def get_prompt(method, query, top_passages_str, last_round_answer=None, **kwargs):
    """Return the appropriate user prompt based on the expansion method."""  
    user_prompts = {
        "thinkqe_revise_0": f"""Given a query and the provided passages (most of which may be incorrect or irrelevant), identify helpful information from the passages and use it to write a correct answering passage. Use your own knowledge, not just the example passages!
        
Query: 
{query}

Possible helpful passages: 
{top_passages_str}
""",
        "thinkqe_revise": f"""Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answer, identify helpful information from the passages and refine the prior answer. 
Ensure the output directly addresses the original query. Use your own knowledge, not just the example passages!

Query: 
{query}

Possible helpful passages: 
{top_passages_str}

Prior generated answer/revised query: 
{last_round_answer}
""",}


    return user_prompts[method]


# def get_prompt(method, query, top_passages_str, **kwargs):
#     """Return the appropriate user prompt based on the expansion method."""  
#     user_prompts = {
#     "thinkqe": f"""Given a question \"{query}\" and its possible answering passages (most of these passages are wrong) enumerated as:
# {top_passages_str}

# please write a correct answering passage. Use your own knowledge, not just the example passages!""",
#     }

#     return user_prompts[method]



