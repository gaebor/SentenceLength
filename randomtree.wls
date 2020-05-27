#!/usr/bin/env wolframscript
ReadStats[filename_, max_: 50] := Module[{x, y, ii},
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

If[Length[$ScriptCommandLine] < 2, 
    Print["Usage: datasetname x_max tol eta maxiter"];
    Exit[1]
  ]

vars = Prepend[ToExpression[$ScriptCommandLine[[3;;]]], $ScriptCommandLine[[2]]];
tol = If[Length[vars]<3, 10^-3, vars[[3]]];
eta = If[Length[vars]<4, 0.1, vars[[4]]];
maxiter = If[Length[vars]<5, 1000, vars[[5]]];

{X, Y} = ReadStats@@If[Length[vars] <= 2, vars, vars[[;;2]]];
max = Max[X];

(*Precompute powers*)
P1 = Module[{a=CoefficientArrays[(p10 x+p11 y + p12 z)^Range[max],{x,y,z}]},
            Table[a[[i+1]][[i]], {i,1,max}]
            ];
P2 = Module[{a=CoefficientArrays[(p200 x^2+p201 x y + p220 x z+ p211 y^2+p212 y z+ p222 z^2)^Range[max/2],{x,y,z}]},
            Table[a[[2i+1]][[i]], {i,1,max/2}]
            ];

ExtractElem[L_,i_] := Part@@Prepend[i,L]
GetCoeff1[i_,j_,k_] := ExtractElem[P1[[i+j+k]], Join[ConstantArray[1,i], ConstantArray[2,j], ConstantArray[3,k]]]
GetCoeff2[i_,j_,k_] := ExtractElem[P2[[(i+j+k)/2]], Join[ConstantArray[1,i], ConstantArray[2,j], ConstantArray[3,k]]]

f1[n0_,n1_,n2_] := If[n1>=1,
    If[n0==1,
        1/n1 GetCoeff1[1,n1-1,0],
        (1/(n1*n2))*Sum[Sum[(i10 + i11)*GetCoeff1[-i10, -i11 + n1, i10 + i11]*GetCoeff2[i10 + n0, -1 + i11, -i10 - i11 + n2], {i11, 1 - i10, Min[n1, -i10 + n2]}], {i10, Max[-n0, -n1], 0}]
      ],
    0]
f2[n0_,n1_,n2_] := If[n2 >= 1 && n2 == -1 + n0,
    If[n1 == 0, 
        (1/n2)*GetCoeff2[n0, 0, n2 - 1], 
        (1/(n1*n2))*Sum[Sum[i11*GetCoeff1[-i10, n1 - i11, i10 + i11]*GetCoeff2[i10 + n0, i11, n2 - 1 - i10 - i11], {i11, -i10, Min[n1, n2 - 1 - i10]}], {i10, Max[-n0, -n1], 0}]
      ],
    0]
f[n_] := If[n == 1, t0, 0] + Sum[t1*f1[n0, 1 + n - 2*n0, -1 + n0] + t2*f2[n0, 1 + n - 2*n0, -1 + n0], {n0, 1, (1 + n)/2}]

Fsym=Table[Expand[f[i]], {i, X}];
F[{t0_, t1_, t2_}, {p10_, p11_, p12_}, {p200_, p201_, p220_, p211_, p212_, p222_}] := Evaluate[Fsym]
DFsym = D[Fsym, {{t0, t1, t2, p10, p11, p12, p200, p201, p220, p211, p212, p222}}];
Jac[{t0_, t1_, t2_}, {p10_, p11_, p12_}, {p200_, p201_, p220_, p211_, p212_, p222_}] := Evaluate[DFsym]
ClearAll[Fsym, DFsym];

SoftMax[x_] := (#/Total[#])&@Exp[x]
DSoftMax[fx_] := DiagonalMatrix[fx] - Outer[Times, fx, fx]
NormalizeSoftMax[l_] := l - Log[Total[Exp[l]]]

learned = vars[[1]] <> ".tree.learned"
x = Log[LoadModel[learned]];
If[Head[x] =!= List || Length[x] != 12, x = ConstantArray[0.0, 12]];

PrintLine[{"", "iter", "objective", "error"}, Streams["stderr"][[1]]];

For[i = 1, i <= maxiter , i++,
    tt = SoftMax[x[[;;3]]];
    pp = SoftMax[x[[4;;6]]];
    ppp = SoftMax[x[[7;;]]];
    predicted = F[tt, pp, ppp];
    obj = Y.Log[Y/predicted];
    DG=ArrayFlatten[{{DSoftMax[tt], 0, 0}, {0, DSoftMax[pp], 0}, {0, 0, DSoftMax[ppp]}}];
    grad = ((-Y/predicted).Jac[tt, pp, ppp]).DG;
    x -= eta*grad;
    x[[;;3]] = NormalizeSoftMax[x[[;;3]]];
    x[[4;;6]] = NormalizeSoftMax[x[[4;;6]]];
    x[[7;;]] = NormalizeSoftMax[x[[7;;]]];
    PrintLine[{"", i, obj, Max[Abs[grad]]}, Streams["stderr"][[1]]];
    If[Not[Head[obj] === Real], Exit[1]];
    If[Max[Abs[grad]] < tol, Break[]];
  ];

f = OpenWrite[learned];
WriteString[f, ToString[Exp[x]]];
Close[f];
