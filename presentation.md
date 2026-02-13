---
dimension: 16:9
css: style.css
---

{#title}
# oh-camel!

A deep dive into one of the most influential languages in computer science.

{pause}

*From type theory to multicore runtime — everything you need to know about OCaml.*

{pause up=sec1}

{#sec1}
# History & Philosophy

{pause}

{.definition title="1973 — The Birth of ML"}
> Robin Milner creates **ML** at the University of Edinburgh
> as the scripting language for the **LCF** theorem prover.
>
> Key insight: a language for *proving theorems* needs a type system
> strong enough to prevent nonsense proofs.

{pause}

{.definition title="1983 — Standard ML"}
> ML is formalized into **Standard ML** — first language with
> formal operational semantics, Hindley-Milner type inference,
> parametric polymorphism, and algebraic data types.

{pause}

{.definition title="1996 — Objective Caml"}
> Xavier Leroy releases **Objective Caml** at INRIA:
> native code compiler, structural object system, module system with functors.
> Renamed to **OCaml** in 2011.

{pause up=hm}

{#hm}
## The Hindley-Milner Heritage

The type inference algorithm at OCaml's core was discovered independently three times:

1. **Haskell Curry** (1958) — Combinatory logic types
2. **Roger Hindley** (1969) — Principal type theorem
3. **Robin Milner** (1978) — Algorithm W for ML

{pause}

{.theorem title="Principal Type Property"}
> For every well-typed expression $e$, there exists a **most general**
> (principal) type $\sigma$ such that every other valid type for $e$
> is a substitution instance of $\sigma$.

{pause}

OCaml **never needs type annotations**. The compiler always infers
the most general type. Not merely `auto` from C++ — a *mathematical guarantee*.

{pause up=design}

{#design}
## Design Philosophy

| Property | OCaml | Haskell | Rust | Java |
|----------|-------|---------|------|------|
| Type inference | Full HM | Full HM | Partial | None |
| Purity | Impure | Pure | Impure | Impure |
| Evaluation | Strict | Lazy | Strict | Strict |
| GC | Yes | Yes | No | Yes |
| Compilation | Native | Native | Native | JIT |
| Module system | Functors | Type classes | Traits | Interfaces |

{pause}

OCaml's bet: the strongest practical type system + imperative escape hatches + predictable performance.

{pause up=sec2}

{#sec2}
# The Core Type System

{pause}

## Primitive Types & Type Inference

```ocaml
(* OCaml infers all of these — no annotations needed *)
let x = 42              (* val x : int *)
let pi = 3.14159        (* val pi : float *)
let greeting = "hello"  (* val greeting : string *)
let flag = true          (* val flag : bool *)
let unit_val = ()        (* val unit_val : unit *)
```

{pause}

`int` and `float` are **not** interchangeable — no implicit coercion:

```ocaml
let bad = 1 + 2.0           (* TYPE ERROR *)
let good = float_of_int 1 +. 2.0   (* OK *)
```

{pause up=products}

{#products}
## Algebraic Data Types: Products

**Tuples** — anonymous products:

```ocaml
let point = (3.0, 4.0)        (* float * float *)
let (x, y) = point             (* destructuring *)
```

{pause}

**Records** — named products with functional update:

```ocaml
type person = {
  name : string;
  age  : int;
  email : string option;
}

let alice = { name = "Alice"; age = 30; email = None }
let older = { alice with age = 31 }
```

{pause up=variants}

{#variants}
## Algebraic Data Types: Sums

```ocaml
type shape =
  | Circle of float
  | Rectangle of float * float
  | Triangle of float * float * float

let area = function
  | Circle r -> Float.pi *. r *. r
  | Rectangle (w, h) -> w *. h
  | Triangle (a, b, c) ->
    let s = (a +. b +. c) /. 2.0 in
    Float.sqrt (s *. (s -. a) *. (s -. b) *. (s -. c))
```

{pause up=poly}

{#poly}
## Parametric Polymorphism

```ocaml
let id x = x                    (* val id : 'a -> 'a *)
let make_pair x y = (x, y)      (* val make_pair : 'a -> 'b -> 'a * 'b *)

let rec length = function
  | [] -> 0
  | _ :: tl -> 1 + length tl    (* val length : 'a list -> int *)
```

{pause}

`'a` is a **type variable**: $\forall \alpha. \; \alpha \to \alpha$

**Parametric** means the function cannot inspect the type at runtime.
**Free theorem**: any function `'a -> 'a` *must* be the identity (or diverge).

{pause up=rectypes}

{#rectypes}
## Recursive Types

```ocaml
type 'a list =
  | []                     (* empty *)
  | (::) of 'a * 'a list   (* head :: tail *)

type 'a tree =
  | Leaf
  | Node of 'a tree * 'a * 'a tree
```

{pause up=gadts}

{#gadts}
## GADTs

Generalized Algebraic Data Types — types that refine based on constructor:

```ocaml
type _ expr =
  | Int  : int -> int expr
  | Bool : bool -> bool expr
  | Add  : int expr * int expr -> int expr
  | If   : bool expr * 'a expr * 'a expr -> 'a expr
  | Eq   : int expr * int expr -> bool expr
```

{pause}

```ocaml
(* Type-safe evaluator — no runtime type errors possible *)
let rec eval : type a. a expr -> a = function
  | Int n -> n
  | Bool b -> b
  | Add (l, r) -> eval l + eval r
  | If (cond, t, e) -> if eval cond then eval t else eval e
  | Eq (l, r) -> eval l = eval r

(* eval (Add (Int 1, Bool true)) — WON'T COMPILE *)
```

{pause up=sec3}

{#sec3}
# Pattern Matching

{pause}

```ocaml
let describe n = match n with
  | 0 -> "zero"
  | 1 -> "one"
  | n when n < 0 -> "negative"
  | _ -> "something else"
```

{pause}

## Exhaustiveness Checking

```ocaml
type traffic_light = Red | Yellow | Green

let action = function
  | Red -> "stop"
  | Green -> "go"
(* Warning 8: not exhaustive — missing: Yellow *)
```

Not a linter — **semantic analysis**. With `-warn-error +8`, this becomes a compile error.

{pause up=nested}

{#nested}
## Deep & Nested Patterns

An expression simplifier — pattern matching at its best:

```ocaml
type expr = Const of int | Add of expr * expr
           | Mul of expr * expr | Neg of expr

let rec simplify = function
  | Add (Const 0, e) | Add (e, Const 0) -> simplify e
  | Mul (Const 1, e) | Mul (e, Const 1) -> simplify e
  | Mul (Const 0, _) | Mul (_, Const 0) -> Const 0
  | Neg (Neg e) -> simplify e
  | Add (Const a, Const b) -> Const (a + b)
  | Neg (Const n) -> Const (-n)
  | Add (l, r) -> Add (simplify l, simplify r)
  | Mul (l, r) -> Mul (simplify l, simplify r)
  | e -> e
```

{pause up=orpat}

{#orpat}
## Or-Patterns, Guards & Exception Patterns

```ocaml
let is_vowel = function
  | 'a' | 'e' | 'i' | 'o' | 'u' -> true
  | _ -> false

let classify_age = function
  | n when n < 13 -> "child"
  | n when n < 20 -> "teenager"
  | n when n < 65 -> "adult"
  | _ -> "senior"
```

{pause}

```ocaml
(* Exception patterns — match and catch in one *)
let safe_divide x y =
  match x / y with
  | result -> Some result
  | exception Division_by_zero -> None
```

{pause up=sec4}

{#sec4}
# The Module System

*Modules are to types what types are to values.*

{pause}

## Structures & Signatures

```ocaml
module type SET = sig
  type elt
  type t
  val empty : t
  val add : elt -> t -> t
  val mem : elt -> t -> bool
end

module IntSet : SET with type elt = int = struct
  type elt = int
  type t = int list
  let empty = []
  let add x s = if List.mem x s then s else x :: s
  let mem = List.mem
end
(* IntSet.t is abstract — callers can't see it's a list *)
```

{pause up=functors}

{#functors}
## Functors: Parameterized Modules

```ocaml
module type COMPARABLE = sig
  type t
  val compare : t -> t -> int
end
```

{pause up=functors2}

{#functors2}

```ocaml
module MakeSet (Elt : COMPARABLE) : SET with type elt = Elt.t =
struct
  type elt = Elt.t
  type t = elt list
  let empty = []
  let rec add x = function
    | [] -> [x]
    | hd :: _ as l when Elt.compare x hd = 0 -> l
    | hd :: tl -> hd :: add x tl
  let mem x = List.exists (fun e -> Elt.compare x e = 0)
end

module StringSet = MakeSet(String)
```

{pause up=fcm}

{#fcm}
## First-Class Modules

```ocaml
(* Pack a module into a value *)
let set_mod : (module SET with type elt = int) = (module IntSet)

(* The stdlib Map is a functor *)
module StringMap = Map.Make(String)

let env =
  StringMap.empty
  |> StringMap.add "HOME" "/home/user"
  |> StringMap.add "PATH" "/usr/bin"
```

{pause up=sec5}

{#sec5}
# Immutability & FP Idioms

{pause}

```ocaml
(* Immutable by default *)
let x = 42            (* "let x = 43" is shadowing, not mutation *)
let xs = [1; 2; 3]
let ys = 0 :: xs      (* [0;1;2;3] — xs unchanged, structural sharing *)
```

{pause}

## The Pipe Operator

```ocaml
let result =
  [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]
  |> List.filter (fun x -> x mod 2 = 0)
  |> List.map (fun x -> x * x)
  |> List.fold_left (+) 0
(* 220 *)
```

{pause up=curry}

{#curry}
## Currying & Closures

Every function in OCaml takes exactly one argument. Multi-argument functions are syntactic sugar:

```ocaml
let add x y = x + y
(* is sugar for: *)
let add = fun x -> fun y -> x + y
(* val add : int -> int -> int *)

let add5 = add 5    (* int -> int — partial application *)
```

{pause up=tailrec}

{#tailrec}
## Tail Recursion & CPS

```ocaml
(* NOT tail recursive — will stack overflow on large lists *)
let rec sum_bad = function
  | [] -> 0
  | x :: xs -> x + sum_bad xs

(* Tail recursive with accumulator *)
let sum lst =
  let rec aux acc = function
    | [] -> acc
    | x :: xs -> aux (acc + x) xs
  in aux 0 lst
```

{pause up=cps}

{#cps}

**Continuation-passing style** — convert any recursion to tail-recursive:

```ocaml
type 'a tree = Leaf | Node of 'a tree * 'a * 'a tree

let map_tree f tree =
  let rec aux t k = match t with
    | Leaf -> k Leaf
    | Node (l, v, r) ->
      aux l (fun l' ->
        aux r (fun r' ->
          k (Node (l', f v, r'))))
  in aux tree Fun.id
```

{pause up=sec6}

{#sec6}
# Imperative Features

*OCaml is pragmatically impure — mutation when you need it.*

{pause}

## References, Mutable Fields & Arrays

```ocaml
(* References: heap-allocated mutable cell *)
let counter = ref 0
let incr () = counter := !counter + 1; !counter
(* ref = { mutable contents : 'a }
   !   = dereference
   :=  = assign *)

(* Mutable record fields *)
type buf = { mutable data: bytes; mutable len: int }

(* Arrays: fixed-size, mutable, O(1) access *)
let arr = [| 1; 2; 3 |]
let () = arr.(0) <- 99
```

{pause up=loops}

{#loops}
## Loops & Hash Tables

```ocaml
let sum_arr a =
  let t = ref 0 in
  for i = 0 to Array.length a - 1 do t := !t + a.(i) done;
  !t

let tbl = Hashtbl.create 16
let () = Hashtbl.replace tbl "key" 42
```

{pause up=seqs}

{#seqs}
## Lazy Sequences

```ocaml
let naturals =
  let rec f n () = Seq.Cons (n, f (n+1)) in f 0

let first_5_even_squares =
  naturals
  |> Seq.filter (fun n -> n mod 2 = 0)
  |> Seq.map (fun n -> n * n)
  |> Seq.take 5
  |> List.of_seq
(* [0; 4; 16; 36; 64] *)
```

{pause up=sec7}

{#sec7}
# Error Handling

{pause}

## Option & Result

```ocaml
(* type 'a option = None | Some of 'a *)
let safe_div x y = if y = 0 then None else Some (x / y)

(* type ('a,'b) result = Ok of 'a | Error of 'b *)
let ( let* ) = Result.bind

let parse_point s =
  match String.split_on_char ',' s with
  | [x; y] ->
    let* x = int_of_string_opt (String.trim x)
             |> Option.to_result ~none:"bad x" in
    let* y = int_of_string_opt (String.trim y)
             |> Option.to_result ~none:"bad y" in
    Ok (x, y)
  | _ -> Error "expected x,y"
```

{pause up=exn}

{#exn}
## Exceptions

```ocaml
exception Config_error of string

(* Exceptions are FAST — setjmp/longjmp, not stack unwinding *)
let require_env var =
  match Sys.getenv_opt var with
  | Some v -> v
  | None -> raise (Config_error ("missing: " ^ var))
```

{pause up=bindops}

{#bindops}
## Binding Operators

```ocaml
(* Monadic syntax for any type (OCaml 4.08+) *)
module Option_syntax = struct
  let ( let* ) = Option.bind
  let ( let+ ) x f = Option.map f x
end

let open Option_syntax in
let+ x = safe_div 10 3 in x * 2
(* Some 6 *)
```

{pause up=sec8}

{#sec8}
# Memory Model & GC

{pause}

{.definition title="Value Representation"}
> Every OCaml value is either an **immediate** (tagged int, LSB=1)
> or a **pointer** to a heap block (LSB=0).
>
> `42` is stored as `(42 << 1) | 1 = 85`. No allocation.
> `3.14`, `"hello"`, `(1,2)` — heap-allocated blocks.

{pause}

## Heap Block Layout

```
+--------+--------+--------+
| header | field0 | field1 | ...
+--------+--------+--------+
header = [ size (22 bits) | color (2 bits) | tag (8 bits) ]
```

Tag: `0` = tuple/record, `252` = string, `253` = float

{pause up=gc}

{#gc}
## The Generational Garbage Collector

{.block title="Minor Heap (~256KB)"}
> Bump-pointer allocation. ~3 instructions, ~10ns per allocation.
> Collected by stop-and-copy GC — most objects die young.

{.block title="Major Heap"}
> Incremental mark-and-sweep with compaction.
> Runs concurrently with the program in small slices.

{pause up=gcapi}

{#gcapi}

```ocaml
let () =
  Gc.set { (Gc.get ()) with
    minor_heap_size = 512 * 1024;
    space_overhead = 80 };
  let s = Gc.stat () in
  Printf.printf "Minor: %d, Major: %d, Heap: %d words\n"
    s.minor_collections s.major_collections s.heap_words
```

Mutation triggers a **write barrier** — if a new value is young and the record is old,
the record enters the "remembered set" so the minor GC can find cross-generational pointers.

{pause up=sec9}

{#sec9}
# The Compiler Pipeline

{pause}

```
Source  →  Parsetree (untyped AST)
       →  Typedtree (typed AST, inference via Algorithm W)
       →  Lambda IR (pattern match → decision trees)
       →  Flambda  (inlining, specialization, unboxing)
       →  Cmm      (C-- low-level IR)
       →  Assembly → Native binary
```

{pause}

## Algorithm W in Action

```ocaml
let f x = x + 1
(* 1. Fresh type: x : 'a                        *)
(* 2. (+) : int -> int -> int, so unify 'a = int *)
(* 3. Result: f : int -> int                     *)
```

Pattern matches compile to **decision trees** — the compiler minimizes
the number of tests by sharing common prefixes.

{pause up=flambda}

{#flambda}
## Flambda & Compilation Modes

| Mode | Command | Performance |
|------|---------|-------------|
| Bytecode | `ocamlc` | Interpreted, fast compile |
| Native | `ocamlopt` | ~10x faster runtime |
| Flambda | `ocamlopt -O2` | Best optimization |

{pause}

Inspect the compiler's work:

```bash
ocamlopt -dlambda file.ml    # Lambda IR output
ocamlopt -dcmm file.ml      # C-- IR output
ocamlopt -S file.ml          # Generated assembly
```

{pause up=sec10}

{#sec10}
# OCaml 5: Multicore & Effects

{pause}

{.block title="Before OCaml 5"}
> Global Interpreter Lock. One domain at a time. Parallelism only via multiprocess.

{.block title="OCaml 5.0 (December 2022)"}
> True shared-memory parallelism via **domains** + **algebraic effects**.
> A 7+ year research and engineering effort.

{pause up=domains}

{#domains}
## Domains: True Parallelism

```ocaml
let fib n =
  let rec f n = if n < 2 then n else f (n-1) + f (n-2) in f n

let () =
  let d1 = Domain.spawn (fun () -> fib 42) in
  let d2 = Domain.spawn (fun () -> fib 40) in
  Printf.printf "%d, %d\n" (Domain.join d1) (Domain.join d2)
(* Both run on separate OS threads — true parallelism *)
```

{pause up=effects}

{#effects}
## Algebraic Effects: Resumable Exceptions

```ocaml
open Effect
open Effect.Deep

type _ Effect.t += Ask : string -> string Effect.t

let greeter () =
  let name = perform (Ask "Name?") in
  Printf.printf "Hello, %s!\n" name
```

{pause up=effects2}

{#effects2}

```ocaml
let () = match_with greeter ()
  { retc = Fun.id; exnc = raise;
    effc = fun (type a) (e : a Effect.t) -> match e with
      | Ask prompt -> Some (fun (k : (a,_) continuation) ->
          print_string prompt;
          continue k (input_line stdin))
      | _ -> None }
```

{pause up=sched}

{#sched}
## Green-Thread Scheduler with Effects

```ocaml
type _ Effect.t +=
  | Yield : unit Effect.t
  | Fork  : (unit -> unit) -> unit Effect.t

let run main =
  let q = Queue.create () in
  let next () = if Queue.is_empty q then ()
                else (Queue.pop q) () in
  let rec go f = match_with f ()
    { retc = (fun () -> next ());
      exnc = raise;
      effc = fun (type a) (e : a Effect.t) -> match e with
        | Yield -> Some (fun k ->
            Queue.push (fun () -> continue k ()) q; next ())
        | Fork f -> Some (fun k ->
            Queue.push (fun () -> continue k ()) q; go f)
        | _ -> None }
  in go main
```

{pause up=sched2}

{#sched2}

```ocaml
let () = run (fun () ->
  perform (Fork (fun () ->
    for i = 1 to 5 do Printf.printf "A%d " i; perform Yield done));
  for i = 1 to 5 do Printf.printf "B%d " i; perform Yield done)
(* A1 B1 A2 B2 A3 B3 A4 B4 A5 B5 *)
```

{pause up=sec11}

{#sec11}
# The Ecosystem

{pause}

## Getting Started

```bash
opam init && opam switch create 5.2.0
opam install dune merlin ocaml-lsp-server ocamlformat
```

{pause}

## Dune Build System

```lisp
; dune-project
(lang dune 3.16)
(name my_project)

; lib/dune
(library (name my_lib) (libraries unix)
 (preprocess (pps ppx_deriving.show)))

; bin/dune
(executable (name main) (libraries my_lib))
```

{pause up=ppx}

{#ppx}
## PPX: Compile-Time Metaprogramming

```ocaml
type point = { x : float; y : float }
[@@deriving show, eq, ord]
(* Generates: show_point, equal_point, compare_point *)

type config = { host: string; port: int }
[@@deriving yojson]
(* Generates: config_of_yojson, yojson_of_config *)
```

{pause up=sec12}

{#sec12}
# Real-World OCaml

{pause}

**Jane Street** — ~30M lines of OCaml powering quantitative trading.
Core/Base, Async, Incremental, Bonsai, ppx_jane.

```ocaml
open Core

let process data =
  data
  |> List.filter ~f:(fun x -> x > 0)
  |> List.map ~f:(fun x -> Float.of_int x |> Float.sqrt)
  |> List.fold ~init:0.0 ~f:(+.)
```

{pause}

**Tezos** — blockchain written in OCaml. Type safety prevents smart contract bugs.

**MirageOS** — library OS (unikernel). ~15MB images, boots in <100ms.

**Also built with OCaml**: Coq, Flow, Hack, Frama-C, Semgrep.

{pause}

{.theorem title="The OCaml Value Proposition"}
> **Correctness** — types catch bugs at compile time.
> **Performance** — native code competitive with C.
> **Expressiveness** — pattern matching + modules + FP.
> **Practicality** — impure when needed, pure when wanted.

{pause up=thanks}

{#thanks}
# Thank You

**Resources**

- [Real World OCaml](https://dev.realworldocaml.org) — free online book
- [ocaml.org](https://ocaml.org) — official docs & tutorials
- [try.ocaml.org](https://try.ocaml.org) — browser playground
- [blog.janestreet.com](https://blog.janestreet.com) — technical blog

```ocaml
let () = print_endline "Happy hacking!"
```
