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
HesseStep[hessian_, grad_, epsilon_: 0.1] := MatrixFunction[If[Abs[#] < epsilon, #/epsilon^2, 1/#]&, hessian].grad

stderr = Streams["stderr"][[1]];

PrintF[x_]:=ToString[NumberForm[x, NumberFormat -> (If[#3 == "", #1, SequenceForm[#1, "E", #3]] &)]]

LoadModel[filename_]:= Module[{x,t},
    {x, y} = Check[ToExpression[ReadString[filename]], {{},{}}];
    If[Head[x] =!= List || Length[x] != 9 || Head[y] != Association || Length[y] == 0, 
        x = ConstantArray[0.0, 9]; y = <| {0,0,1}->1 |>,
        Write[stderr, "Loaded \"" <> filename <> "\""]
    ];
    {x, y}
  ]

MyTiming[L_, message_:""]:=Module[{time},
    time = AbsoluteTime[];
    WriteString[stderr, message];
    ReleaseHold[L];
    Write[stderr, PrintF[AbsoluteTime[]-time], " sec, ", PrintF[N[MemoryInUse[]]], " byte"];
]
If[Length[$ScriptCommandLine] < 2, 
    Print["Usage: datasetname x_max tol eta maxiter"];
    Exit[1]
  ]

vars = Prepend[ToExpression[$ScriptCommandLine[[3;;]]], $ScriptCommandLine[[2]]];
tol = If[Length[vars]<3, 10^-3, vars[[3]]];
eta = If[Length[vars]<4, 0.1, vars[[4]]];
maxiter = If[Length[vars]<5, 1000, vars[[5]]];
usehessian = If[Length[vars]<6, 0, vars[[6]]];

{X, Y} = ReadStats@@If[Length[vars] <= 2, vars, vars[[;;2]]];
max = Max[X];

MyTiming[
  Hold[
    P1 = Module[{a=CoefficientArrays[(p10 x+p11 y + p12 z)^Range[max],{x,y,z}]},
            Table[a[[i+1]][[i]], {i,1,max}]
            ];
    P2 = Module[{a=CoefficientArrays[(p200 x^2+p201 x y + p220 x z+ p211 y^2+p212 y z+ p222 z^2)^Range[max/2],{x,y,z}]},
            Table[a[[2i+1]][[i]], {i,1,max/2}]
            ]
  ],
  "Precompute powers ... "
];

ExtractElem[L_,i_] := Part@@Prepend[i,L]
GetCoeff1[0,0,0] = 1;
GetCoeff2[0,0,0] = 1;
GetCoeff1[i_,j_,k_] := ExtractElem[P1[[i+j+k]], Join[ConstantArray[1,i], ConstantArray[2,j], ConstantArray[3,k]]]
GetCoeff2[i_,j_,k_] := ExtractElem[P2[[(i+j+k)/2]], Join[ConstantArray[1,i], ConstantArray[2,j], ConstantArray[3,k]]]

fk[{k0_, k1_, k2_}, {n0_, n1_, n2_}] := 
 Module[{nnz = Flatten[Position[{n0, n1, n2}, _?Positive]], kappa=k0 + k1 + k2},
  If[n0 - n2 == kappa && n0 >= k0 && n1 >= k1 && n2 >= k2,
   (1/(Times @@ ({n0, n1, n2}[[nnz]])))*Sum[
     Sum[
        Det[{{n0, 0, 0}, {i10, i11, -i10 - i11}, {-i10 + k0 - n0, -i11 + k1, i10 + i11 + k2}}[[nnz, nnz]]]*
        GetCoeff1[-i10, n1 - i11, i10 + i11]*GetCoeff2[i10 - k0 + n0, i11 - k1, n2 - (i10 + i11 + k2)], 
      {i11, Max[k1, -i10], Min[n1, n2 - k2 - i10]}
      ],
    {i10, Max[k0 - n0, -n1], 0}
    ],
   0]
  ];

f[k_, n_] := Sum[fk[k, {n0, Total[k] + n - 2*n0, n0 - Total[k]}], {n0, Total[k], (Total[k] + n)/2}]

learned = vars[[1]] <> ".tree.learned"
{x, mixture} = LoadModel[learned];
x = Log[Join[x, Values[mixture]]];

MyTiming[Hold[
    F[{p10_, p11_, p12_}, {p200_, p201_, p220_, p211_, p212_, p222_}] := 
        Evaluate[Table[Expand[f[k, n]], {k, Keys[mixture]}, {n, X}]];
    ], "Calculating prediction ... "];

MyTiming[Hold[
    DF[{p10_, p11_, p12_}, {p200_, p201_, p220_, p211_, p212_, p222_}] := 
        Evaluate[D[
            F[{p10, p11, p12},{p200, p201, p220, p211, p212, p222}],
            {{p10, p11, p12, p200, p201, p220, p211, p212, p222}}
            ]]
        ], "Calculating derivative ... "];

(*MyTiming[Hold[
    DDFsym = Table[D[df, {{t0, t1, t2, p10, p11, p12, p200, p201, p220, p211, p212, p222}}], {df, DFsym}];
    DDF[{t0_, t1_, t2_}, {p10_, p11_, p12_}, {p200_, p201_, p220_, p211_, p212_, p222_}] := Evaluate[DDFsym]],
"Calculating Hessian ... "];*)

SoftMax[x_] := (#/Total[#])&@Exp[x]
DSoftMax[fx_] := DiagonalMatrix[fx] - Outer[Times, fx, fx]
NormalizeSoftMax[l_] := l - Log[Total[Exp[l]]]

Write[stderr, "\titer\tobjective\terror\ttime[sec]\tmemory[byte]"];

For[i = 1, i <= maxiter , i++,
    time = AbsoluteTime[];
    x[[;;3]] = NormalizeSoftMax[x[[;;3]]];
    x[[4;;9]] = NormalizeSoftMax[x[[4;;9]]];
    x[[10;;]] = NormalizeSoftMax[x[[10;;]]];
    pp = Exp[x[[;;3]]];
    ppp = Exp[x[[4;;9]]];
    tt = Exp[x[[10;;]]];
    Fnum = F[pp, ppp];
    denom = Y/tt.Fnum;
    obj = Y.Log[denom];
    DG=ArrayFlatten[{{DSoftMax[tt], 0, 0}, {0, DSoftMax[pp], 0}, {0, 0, DSoftMax[ppp]}}];
    grad = (-denom.Join[tt.DF[pp, ppp], Transpose[Fnum], 2]).DG;
    x -= eta*grad;
    logdata = {i, obj, Max[Abs[grad]], AbsoluteTime[]-time, N[MemoryInUse[]]};
    Write@@Riffle[
        Prepend[PrintF/@logdata, stderr], 
        ConstantArray["\t", Length[logdata]]
        ];
    If[Not[Head[obj] === Real], Exit[1]];
    If[Max[Abs[grad]] < tol, Break[]];
  ];

f = OpenWrite[learned];
WriteString[f, ToString[{
    Join[SoftMax[x[[;;3]]], SoftMax[x[[4;;9]]]],
    Association@@Map[Rule@@#&, Transpose@{Keys[mixture], SoftMax[x[[10;;]]]}]
    }]
];
Close[f];

(*Print[F[x[[;;3]], x[[4;;]]]]*)
