---
dimension: 16:9
css: style.css
---

{#title}
# OCaml: A Deep Technical Dive

**From type theory to multicore runtime**

{pause}

A comprehensive journey through one of the most influential
programming languages in computer science.

{pause up=sec1}

{#sec1}
# 1. History and Philosophy

{pause}

{.block title="1973: The Birth of ML"}
> Robin Milner creates **ML** (Meta Language) at the University of Edinburgh
> as the scripting language for the **LCF** theorem prover.
>
> Key insight: a language designed for *proving theorems* needs a type system
> strong enough to prevent nonsense proofs.

{pause}

{.block title="1983: Standard ML"}
> ML is formalized into **Standard ML** — first language with:
>
> - Formal operational semantics
> - Hindley-Milner type inference
> - Parametric polymorphism
> - Algebraic data types

{pause}

{.block title="1996: Objective Caml"}
> Xavier Leroy releases **Objective Caml** at INRIA with:
>
> - Native code compiler
> - Object system (structural typing)
> - Module system with functors
>
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
> For every well-typed expression $e$ in the Hindley-Milner system,
> there exists a **most general** (principal) type $\sigma$ such that
> every other valid type for $e$ is a substitution instance of $\sigma$.

{pause}

**OCaml never needs type annotations.** The compiler can always
infer the most general type. Not merely `auto` from C++ — a *mathematical guarantee*.

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

**OCaml's bet**: strongest practical type system + imperative escape hatches + predictable performance.

{pause up=sec2}

{#sec2}
# 2. The Core Type System

{pause}

## Primitive Types and Type Inference

```ocaml
(* OCaml infers all of these *)
let x = 42              (* val x : int *)
let pi = 3.14159        (* val pi : float *)
let greeting = "hello"  (* val greeting : string *)
let flag = true          (* val flag : bool *)
let unit_val = ()        (* val unit_val : unit *)
```

{pause}

`int` and `float` are **not** interchangeable — no implicit coercion:

```ocaml
let bad = 1 + 2.0   (* TYPE ERROR *)
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

**Records** — named products:

```ocaml
type person = {
  name : string;
  age  : int;
  email : string option;
}

let alice = { name = "Alice"; age = 30; email = None }
let older = { alice with age = 31 }  (* functional update *)
```

{pause up=variants}

{#variants}
## Algebraic Data Types: Sums (Variants)

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

**Parametric** polymorphism — the function cannot inspect the type at runtime.
**Free theorem**: any `'a -> 'a` *must* be the identity (or diverge).

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
## GADTs (Generalized Algebraic Data Types)

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

(* eval (Add (Int 1, Bool true)) — COMPILE TIME ERROR *)
```

{pause up=sec3}

{#sec3}
# 3. Pattern Matching

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
(* Warning 8: not exhaustive. Missing: Yellow *)
```

Not a linter — **semantic analysis**. With `-warn-error +8`, this is a compile error.

{pause up=nested}

{#nested}
## Deep / Nested Patterns

```ocaml
type expr = Const of int | Add of expr * expr
           | Mul of expr * expr | Neg of expr

let rec simplify = function
  | Add (Const 0, e) | Add (e, Const 0) -> simplify e
  | Mul (Const 1, e) | Mul (e, Const 1) -> simplify e
  | Mul (Const 0, _) | Mul (_, Const 0) -> Const 0
  | Neg (Neg e) -> simplify e
  | Add (Const a, Const b) -> Const (a + b)
  | Mul (Const a, Const b) -> Const (a * b)
  | Neg (Const n) -> Const (-n)
  | Add (l, r) -> Add (simplify l, simplify r)
  | Mul (l, r) -> Mul (simplify l, simplify r)
  | Neg e -> Neg (simplify e)
  | Const _ as e -> e
```

{pause up=orpat}

{#orpat}
## Or-Patterns, Guards, and More

```ocaml
let is_vowel = function
  | 'a' | 'e' | 'i' | 'o' | 'u' -> true
  | _ -> false

let classify_age = function
  | n when n < 13 -> "child"
  | n when n < 20 -> "teenager"
  | n when n < 65 -> "adult"
  | _ -> "senior"

(* Exception patterns *)
let safe_divide x y =
  match x / y with
  | result -> Some result
  | exception Division_by_zero -> None
```

{pause up=sec4}

{#sec4}
# 4. The Module System

Modules are to types what types are to values.

{pause}

## Structures and Signatures

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
(* IntSet.t is now abstract — callers can't see it's a list *)
```

{pause up=functors}

{#functors}
## Functors: Parameterized Modules

```ocaml
module type COMPARABLE = sig
  type t
  val compare : t -> t -> int
end

module MakeSet (Elt : COMPARABLE) : SET with type elt = Elt.t =
struct
  type elt = Elt.t
  type t = elt list
  let empty = []
  let rec add x = function
    | [] -> [x]
    | hd :: tl as l ->
      let c = Elt.compare x hd in
      if c = 0 then l
      else if c < 0 then x :: l
      else hd :: add x tl
  let rec mem x = function
    | [] -> false
    | hd :: _ when Elt.compare x hd = 0 -> true
    | _ :: tl -> mem x tl
end

module StringSet = MakeSet(String)
```

{pause up=fcm}

{#fcm}
## First-Class Modules and Map.Make

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
# 5. Immutability and FP Idioms

{pause}

```ocaml
let x = 42            (* immutable — "let x = 43" is shadowing, not mutation *)
let xs = [1; 2; 3]
let ys = 0 :: xs      (* [0;1;2;3] — xs unchanged *)
```

{pause}

## Higher-Order Functions and the Pipe

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
## Closures and Currying

```ocaml
let add x y = x + y           (* int -> int -> int *)
(* sugar for: fun x -> fun y -> x + y *)

let add5 = add 5              (* int -> int — closure *)
let is_even = (fun n x -> x mod n = 0) 2
```

{pause up=tailrec}

{#tailrec}
## Tail Recursion and CPS

```ocaml
(* NOT tail recursive — stack overflow *)
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

{pause}

```ocaml
(* CPS: convert any recursion to tail-recursive *)
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
# 6. Imperative Features

OCaml is pragmatically impure.

{pause}

```ocaml
(* References *)
let counter = ref 0
let incr () = counter := !counter + 1; !counter
(* ref = { mutable contents : 'a }, ! = deref, := = assign *)

(* Mutable record fields *)
type buf = { mutable data: bytes; mutable len: int }

(* Arrays: fixed-size, mutable, O(1) access *)
let arr = [| 1; 2; 3 |]
let () = arr.(0) <- 99
```

{pause up=loops}

{#loops}
## Loops, Hash Tables, Sequences

```ocaml
(* For/while loops *)
let sum_arr a =
  let t = ref 0 in
  for i = 0 to Array.length a - 1 do t := !t + a.(i) done;
  !t

(* Hash tables *)
let tbl = Hashtbl.create 16
let () = Hashtbl.replace tbl "key" 42

(* Lazy sequences *)
let naturals =
  let rec f n () = Seq.Cons (n, f (n+1)) in f 0

let squares = naturals
  |> Seq.filter (fun n -> n mod 2 = 0)
  |> Seq.map (fun n -> n * n)
  |> Seq.take 5 |> List.of_seq
(* [0; 4; 16; 36; 64] *)
```

{pause up=sec7}

{#sec7}
# 7. Error Handling

{pause}

## Option and Result

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
## Exceptions and Binding Operators

```ocaml
exception Config_error of string

(* Exceptions are FAST — setjmp/longjmp, not stack unwinding *)
let require_env var =
  match Sys.getenv_opt var with
  | Some v -> v
  | None -> raise (Config_error ("missing: " ^ var))
```

{pause}

```ocaml
(* Binding operators for any monad (OCaml 4.08+) *)
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
# 8. Memory Model and GC

{pause}

{.definition title="Value Representation"}
> Every value is either an **immediate** (tagged int, LSB=1)
> or a **pointer** to a heap block (LSB=0).
>
> `42` is stored as `(42 << 1) | 1 = 85`. No allocation.
> `3.14`, `"hello"`, `(1,2)` are heap-allocated blocks.

{pause}

## Heap Block Layout

```
+--------+--------+--------+
| header | field0 | field1 | ...
+--------+--------+--------+
header = [ size | color | tag ]
```

Tag: 0 = tuple/record, 252 = string, 253 = float

{pause up=gc}

{#gc}
## The Generational GC

{.block title="Minor Heap (256KB)"}
> Bump-pointer allocation (~3 instructions, ~10ns).
> Collected by copying GC.

{.block title="Major Heap"}
> Incremental mark-and-sweep with compaction.

{pause}

```ocaml
let () =
  Gc.set { (Gc.get ()) with
    minor_heap_size = 512 * 1024;
    space_overhead = 80 };
  let s = Gc.stat () in
  Printf.printf "Minor: %d, Major: %d, Heap: %d words\n"
    s.minor_collections s.major_collections s.heap_words
```

Mutation triggers a **write barrier** — if new value is young and record is old, record goes in the "remembered set."

{pause up=sec9}

{#sec9}
# 9. The Compiler Pipeline

{pause}

```
Source → Parsing → Parsetree (untyped AST)
       → Typing  → Typedtree (typed AST)
       → Lambda  → Lambda IR (pattern compilation)
       → Flambda → Optimized IR
       → Cmm     → C-- (low-level IR)
       → Emit    → Assembly → Native binary
```

{pause}

## Type Inference (Algorithm W)

```ocaml
(* let f x = x + 1 *)
(* 1. x : 'a, (+) : int -> int -> int *)
(* 2. Unify 'a = int *)
(* 3. Result: f : int -> int *)
```

Pattern matches compile to **decision trees** minimizing tests.

{pause up=flambda}

{#flambda}
## Flambda and Compilation Modes

Flambda (`-O2`/`-O3`): inlining, specialization, unboxing, dead code elimination.

| Mode | Command | Runtime |
|------|---------|---------|
| Bytecode | `ocamlc` | Interpreted |
| Native | `ocamlopt` | ~10x faster |
| Flambda | `ocamlopt -O2` | Best optimization |

```bash
ocamlopt -dlambda file.ml    # Lambda IR
ocamlopt -dcmm file.ml      # C-- IR
ocamlopt -S file.ml          # Assembly
```

{pause up=sec10}

{#sec10}
# 10. OCaml 5: Multicore and Effects

{pause}

{.block title="Before OCaml 5"}
> Global Interpreter Lock. One domain at a time. Parallelism = multiprocess.

{.block title="OCaml 5.0 (Dec 2022)"}
> True parallelism via **domains** + **algebraic effects**. 7+ year effort.

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
(* Both run on separate OS threads in parallel *)
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

let () = match_with greeter ()
  { retc = Fun.id; exnc = raise;
    effc = fun (type a) (e : a Effect.t) -> match e with
      | Ask p -> Some (fun (k : (a,_) continuation) ->
          print_string p;
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
        | Yield -> Some (fun (k:(a,_) continuation) ->
            Queue.push (fun () -> continue k ()) q; next ())
        | Fork f -> Some (fun (k:(a,_) continuation) ->
            Queue.push (fun () -> continue k ()) q; go f)
        | _ -> None }
  in go main
```

{pause}

```ocaml
let () = run (fun () ->
  perform (Fork (fun () ->
    for i = 1 to 5 do Printf.printf "A%d " i; perform Yield done));
  for i = 1 to 5 do Printf.printf "B%d " i; perform Yield done)
(* A1 B1 A2 B2 A3 B3 A4 B4 A5 B5 *)
```

{pause up=sec11}

{#sec11}
# 11. The OCaml Ecosystem

{pause}

```bash
opam init && opam switch create 5.2.0
opam install dune merlin ocaml-lsp-server
```

{pause}

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

{pause}

## PPX: Compile-Time Metaprogramming

```ocaml
type point = { x : float; y : float }
[@@deriving show, eq, ord]
(* Generates: show_point, equal_point, compare_point *)

type config = { host: string; port: int } [@@deriving yojson]
(* Generates: config_of_yojson, yojson_of_config *)
```

{pause up=sec12}

{#sec12}
# 12. Real-World OCaml

{pause}

**Jane Street** — ~30M lines of OCaml for quantitative trading.
Core/Base, Async, Incremental, Bonsai, ppx_jane.

```ocaml
open Core
let process data =
  data |> List.filter ~f:(fun x -> x > 0)
       |> List.map ~f:(fun x -> Float.of_int x |> Float.sqrt)
       |> List.fold ~init:0.0 ~f:(+.)
```

{pause}

**Tezos** — blockchain in OCaml. Type safety prevents smart contract bugs.

**MirageOS** — library OS (unikernel). ~15MB, boots in <100ms.

**Also**: Coq, Flow, Hack, Frama-C, Semgrep

{pause}

{.theorem title="The OCaml Value Proposition"}
> **Correctness** — types catch bugs at compile time
> **Performance** — native code competitive with C
> **Expressiveness** — pattern matching + modules + FP
> **Practicality** — impure when needed, pure when wanted

{pause up=thanks}

{#thanks}
# Thank You

- *Real World OCaml* — dev.realworldocaml.org (free)
- ocaml.org — official docs
- try.ocaml.org — playground
- blog.janestreet.com — tech blog

```ocaml
let () = print_endline "Happy hacking!"
```
