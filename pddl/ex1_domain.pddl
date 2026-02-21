(define (domain box-push)
  (:requirements :strips :typing)
  (:types agent location box bigbox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (bigbox-at ?b - bigbox ?loc1 - location ?loc2 - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (goal ?loc - location)
    (won )
  )
  
  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)) (not (clear ?to)) (clear ?from))
  )
  
  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and (agent-at ?a ?from) (adj ?from ?boxloc) (box-at ?b ?boxloc) (adj ?boxloc ?toloc) (clear ?toloc))
    :effect (and (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from) (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc)))
  )

  (:action push-big
    :parameters (?a1 - agent ?a2 - agent ?from1 - location ?from2 - location 
                 ?boxloc1 - location ?boxloc2 - location 
                 ?toloc1 - location ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )
  
  (:action win-small
    :parameters (?b - box ?loc - location)
    :precondition (and (box-at ?b ?loc) (goal ?loc))
    :effect (won)
  )

  (:action win-big-1
    :parameters (?b - bigbox ?loc1 - location ?loc2 - location)
    :precondition (and (bigbox-at ?b ?loc1 ?loc2) (goal ?loc1))
    :effect (won)
  )
  
  (:action win-big-2
    :parameters (?b - bigbox ?loc1 - location ?loc2 - location)
    :precondition (and (bigbox-at ?b ?loc1 ?loc2) (goal ?loc2))
    :effect (won)
  )
)
