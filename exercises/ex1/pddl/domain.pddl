(define (domain box-push)
  (:requirements :strips :typing :disjunctive-preconditions)
  (:types agent location box bigbox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (bigbox-at ?b - bigbox ?loc1 - location ?loc2 - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (adj-left ?l1 - location ?l2 - location)
    (adj-right ?l1 - location ?l2 - location)
    (adj-up ?l1 - location ?l2 - location)
    (adj-down ?l1 - location ?l2 - location)
  )
  
  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)) (not (clear ?to)) (clear ?from))
  )
  
  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and 
        (agent-at ?a ?from) 
        (box-at ?b ?boxloc) 
        (clear ?toloc)
        (or 
            (and (adj-up ?from ?boxloc) (adj-up ?boxloc ?toloc))
            (and (adj-down ?from ?boxloc) (adj-down ?boxloc ?toloc))
            (and (adj-left ?from ?boxloc) (adj-left ?boxloc ?toloc))
            (and (adj-right ?from ?boxloc) (adj-right ?boxloc ?toloc))
        )
    )
    :effect (and (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from) (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc)))
  )

  (:action push-big-up
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-up ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-up ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-right ?boxloc1 ?boxloc2)
        (adj-up ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-up ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-down
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-down ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-down ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-right ?boxloc1 ?boxloc2)
        (adj-down ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-down ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-left
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-left ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-left ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-down ?boxloc1 ?boxloc2)
        (adj-left ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-left ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-right
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-right ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-right ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-down ?boxloc1 ?boxloc2)
        (adj-right ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-right ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )
)
