using FacilityLocationProblems

problem = FacilityLocationProblem(20, 500, 2)
solution, _ = local_search(problem)
