#!/usr/bin/env wolframscript

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

PrintF[x_]:=ToString[NumberForm[x, NumberFormat -> (If[#3 == "", #1, SequenceForm[#1, "e", #3]] &)]]

MyPrint[x_] := If[Head[x] === String, x, PrintF[x]]
LoadModel[filename_]:=Check[ToExpression[ReadString[filename]], {}]
PrintLine[x_, stream_: Streams["stdout"][[1]]]:= WriteLine[stream, 
        If[Head[x] === List, StringRiffle[MyPrint/@x, "\t"], MyPrint[x]]
    ]
HesseStep[hessian_, grad_, epsilon_: 0.1] := MatrixFunction[If[#<1/epsilon, epsilon, 1/#]&, hessian].grad
Sichel[a_, b_, r_, g_: -1/2] := ((1 - b)^(g/2)*(a*b/2)^r * BesselK[g + r, a])/(BesselK[g, a*Sqrt[1 - b]]*r!)

Optimize[dataset_, max_: 100, tol_: 10^-3, eta_: 1.0, maxiter_: 1000, usehessian_: False, gamma_: -1/2] :=
Module[{learned=dataset <> ".g" <> ToString[N[gamma]] <> ".learned",
         x0, x, y, i, f, df, ddf, obj, grad, hessian, constant},
  x0 = LoadModel[learned];
  If[Head[x0] =!= List || Length[x0] != 2, x0={10.0, 0.95}];
  {x, y} = ReadStats[dataset, max];
  (*x = x-1;*)
  constant = y.Log[y] + y.Log[x!];
  f[{a_, b_}] := constant - Sum[
        y[[i]] Log[((1 - b)^(gamma/2)*(a*b/2)^x[[i]]*BesselK[gamma + x[[i]], a])/(BesselK[gamma, a*Sqrt[1 - b]])]
    , {i, Length[x]}];
  df = {Derivative[{1, 0}][f], Derivative[{0, 1}][f]};
  ddf = {{Derivative[{1, 0}][df[[1]]], Derivative[{0, 1}][df[[1]]]}, 
         {Derivative[{1, 0}][df[[2]]], Derivative[{0, 1}][df[[2]]]}};
  PrintLine["\titer\tobjective\terror", Streams["stderr"][[1]]];
  For[i = 1, i <= maxiter , i++,
    obj = f[x0];
    grad = {df[[1]][x0], df[[2]][x0]};
    If [usehessian > 0, hessian = {{ddf[[1, 1]][x0], ddf[[1, 2]][x0]}, 
                                   {ddf[[2, 1]][x0], ddf[[2, 2]][x0]}}];
    Switch[usehessian,
        1, x0 -= eta*LinearSolve[hessian, grad],
        2, x0 -= HesseStep[hessian, grad, eta],
        _, x0 -= eta*grad
    ];
    PrintLine[{"", i, obj, Max[Abs[grad]]}, Streams["stderr"][[1]]];
    If[Not[Head[obj] === Real && 0<x0[[1]]<100 && 0<x0[[2]]<1], 
        WriteLine[Streams["stderr"][[1]], ToString[x0]];
        Return[1]
      ];
    If[Max[Abs[grad]] < tol, Break[]];
  ];
  hessian = {{ddf[[1, 1]][x0], ddf[[1, 2]][x0]}, 
             {ddf[[2, 1]][x0], ddf[[2, 2]][x0]}};
  obj = f[N[x0, 19]];
  PrintLine[{obj, " ", 0, " ", Log[100.0], " ", 0, " ", Log[Det[hessian]], " ", 0, " ", 2}];
  f = OpenWrite[learned];
  WriteString[f, ToString[x0]];
  Close[f];
  Return[0]
]

If[Length[$ScriptCommandLine] < 2, 
    Print["Usage: datasetname x_max tol eta maxiter usehessian gamma"];
    Exit[1]
  ]

Exit[Optimize@@Prepend[ToExpression[$ScriptCommandLine[[3;;]]], $ScriptCommandLine[[2]]]]
