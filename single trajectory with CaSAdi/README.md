single traj. with CaSadi: round cycle, cubic bezier curve  
original version: no constraints  
update demo: 

8th May: trial of unconstrained model in straight line

11th May: 1.cubic bezier curve trajectory promoted, with maths model.  
2. naive unconstrained optimization model promoted, with CasADi code and maths model.  
3. trial for basic model with both round traj. and cubic bezier curve


19th MAY: now working on opt with double track and nonlinear tire. updated short version but lot to modify


20h MAY: uplodae trial.M, successfully finding the opt solution with double track plus nolinear tire while no constraints added now. need to add later

20th MAY: uploaded approch1.py. A path planner with obstavle avoidance.

21th MAY：Thanks to GOD. with constraints it finally works. NEXT TO DO: get it work on REAL TRACK as well as handle obstacles.


25th MAY: define simple obstacles and get it involved with whole NLP optimal control problem. Also modify the opt objective function. NEXT to do: implement the code in Casadito see how it works and  handle the abortion problem.


ONE thing to mention: if one build a maths model then simulate on ones own. There is NOTHING like "she does everything based on my work" NOR "all your job is to realize what we did". Its cooperation. Everyone has their own job. I proposed my own idea. 


29th MAY: updated trial.M obstconstraint are inactive seems like some conditions for it cant work. Upadated newwersion.m, obstacle constraints are active as well as hard safety constraints but visualization seems like weird, mb problem with projection. no later than 31st MAY, new maths model will be uploaded and check conditions again.


5th JUNE: do 2 tricks too get solution more smooth:1,segment to get more discrezation points. 2,get a weight of density according to the 2nd derivatives and 1st derivatives of curve. see segment1curvature.m


7th JUNE: write one algorithm for the initials. looks good. see segment1curvature1initials.m


13th JUNE: upload full3.pdf to describe what i have done now.


25th JUNE: keep tuning for best parameter. and plz DM me if i need to do braking too


22nd July in last 4 weeks i am keeping on tuning and modify with better numerical results. now uploaded the final version for most of the visualization results, named as segment1curvature1initialswithmultiple.m
