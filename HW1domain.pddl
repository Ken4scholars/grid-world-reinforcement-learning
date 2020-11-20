(define (domain grid_world ) 
(:requirements :strips :typing) 
(:types car
agent - car
gridcell
timeslot
) 
(:predicates (at ?pt1 - gridcell ?car - car) 
(up_next ?pt1 - gridcell ?pt2 - gridcell) 
(down_next ?pt1 - gridcell ?pt2 - gridcell) 
(forward_next_1 ?pt1 - gridcell ?pt2 - gridcell) 
(forward_next_2 ?pt1 - gridcell ?pt2 - gridcell) 
(forward_next_3 ?pt1 - gridcell ?pt2 - gridcell) 
(now ?t - timeslot) 
(next_time ?t1 - timeslot ?t2 - timeslot) 
(overtake ?pt1 - gridcell ?pt2 - gridcell ?t - timeslot) 
(blocked ?pt1 - gridcell ?t - timeslot) 
) 
(:action UP
:parameters ( ?car - agent ?pt1 - gridcell ?pt2 - gridcell ?t1 - timeslot ?t2 - timeslot) 
:precondition (and (at ?pt1 ?car) (up_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)) (now ?t1) (next_time ?t1 ?t2))
:effect (and (not (at ?pt1 ?car)) (at ?pt2 ?car) (not (now ?t1)) (now ?t2))
) 
(:action DOWN
:parameters ( ?car - agent ?pt1 - gridcell ?pt2 - gridcell ?t1 - timeslot ?t2 - timeslot) 
:precondition (and (at ?pt1 ?car) (down_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)) (now ?t1) (next_time ?t1 ?t2))
:effect (and (not (at ?pt1 ?car)) (at ?pt2 ?car )(not (now ?t1)) (now ?t2))
) 
(:action FORWARD-1
:parameters ( ?car - agent ?pt1 - gridcell ?pt2 - gridcell ?t1 - timeslot ?t2 - timeslot) 
:precondition (and (at ?pt1 ?car) (forward_next_1 ?pt1 ?pt2) (not (blocked ?pt2 ?t2)) (now ?t1) (next_time ?t1 ?t2))
:effect (and (not (at ?pt1 ?car)) (at ?pt2 ?car) (not (now ?t1)) (now ?t2))
) 
(:action FORWARD-2
:parameters ( ?car - agent ?pt1 - gridcell ?pt2 - gridcell ?t1 - timeslot ?t2 - timeslot) 
:precondition (and (at ?pt1 ?car) (forward_next_2 ?pt1 ?pt2) (not (blocked ?pt2 ?t2)) (now ?t1) (next_time ?t1 ?t2))
:effect (and (not (at ?pt1 ?car)) (at ?pt2 ?car) (not (now ?t1)) (now ?t2))
) 
(:action FORWARD-3
:parameters ( ?car - agent ?pt1 - gridcell ?pt2 - gridcell ?t1 - timeslot ?t2 - timeslot) 
:precondition (and (at ?pt1 ?car) (forward_next_3 ?pt1 ?pt2) (not (blocked ?pt2 ?t2)) (not (overtake ?pt1 ?pt2 ?t2)) (now ?t1) (next_time ?t1 ?t2))
:effect (and (not (at ?pt1 ?car)) (at ?pt2 ?car) (not (now ?t1)) (now ?t2))
) 
) 
