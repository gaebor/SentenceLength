#!/usr/local/bin/WolframScript -script

ReadStats[filename_, max_: 100] := Module[{x, y, ii},
  {x, y} = 
   Transpose[
    Map[ToExpression, 
     StringSplit /@ ReadList[filename, String],
      2]];
  ii = Flatten[Position[x, _Integer?(# <= max &)]];
  x = x[[ii]]; y = y[[ii]];
  y /= N[Total[y]];
  {x, y}
  ]

LoadModel[filename_]:=Check[ToExpression[ReadString[filename]], {}]

PrintErr[x_]:= If[Head[x] === List, 
        WriteLine[Streams["stderr"][[1]], StringJoin@@(ToString/@x)], 
        WriteLine[Streams["stderr"][[1]], ToString[x]] 
    ]

Optimize[dataset_, max_: 100, tol_: 10^-3, eta_: 1.0, maxiter_: 1000, usehessian_: False, gamma_: -1/2] :=
 Module[{learned=dataset <> ".g" <> ToString[N[gamma]] <> ".learned",
         x0, x, y, i, f, obj, grad, hessian},
  x0 = LoadModel[learned];
  If[Head[x0] =!= List || Length[x0] != 2, x0={10.0, 0.95}];
  {x, y} = ReadStats[Directory[] <> "/" <> dataset, max];
  f[{a_, b_}] := Sum[
        y[[i]] Log[y[[i]]/(((1 - b)^(gamma/2)*(a*b/2)^x[[i]]*BesselK[gamma + x[[i]], a])/(BesselK[gamma, a*Sqrt[1 - b]]*x[[i]]!))]
    , {i, Length[x]}];
  For[i = 1, i <= maxiter , i++,
   obj = f[x0];
   grad = {Derivative[{1, 0}][f][x0], Derivative[{0, 1}][f][x0]};
   If[usehessian, 
     hessian = {{Derivative[{2, 0}][f][x0], Derivative[{1, 1}][f][x0]}, 
                {Derivative[{1, 1}][f][x0], Derivative[{0, 2}][f][x0]}};
     x0 -= eta*LinearSolve[hessian, grad],
     x0 -= eta*grad
   ];
   PrintErr[{"\t", i, "\t", obj, "\t", Max[Abs[grad]]}];
   If[Max[Abs[grad]] <= tol, Break[]];
   If[Head[obj] =!= Real, Return[1]];
  ];
  hessian = {{Derivative[{2, 0}][f][x0], 
      Derivative[{1, 1}][f][x0]}, {Derivative[{1, 1}][f][x0], 
      Derivative[{0, 2}][f][x0]}};
  obj = f[N[x0, 19]];
  Print[obj, " ", 0, " ", Log[100.0], " ", 0, " ", Log[Det[hessian]], " ", 0, " ", 2];
  f = OpenWrite[learned];
  WriteLine[f, ToString[x0]];
  Close[f];
  Return[0]
 ]

Optimize@@Prepend[ToExpression[$ScriptCommandLine[[3;;]]], $ScriptCommandLine[[2]]]
