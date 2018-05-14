%% IEEEAerospace2015.cls
%%
%% This file was originally based on IEEEtran.sty and modified by
%% Alexander Tartakovsky, Phillip De Leon, Peter Willett, and
%% Slava Turyshev over the years 2002-2005.
%%
%% Grateful thanks to David T�rnqvist <tornqvist@isy.liu.se> for fixing the
%% amsmath package bug 10/2009.
%%
%% Modifications made to conform to the 2013 Aerospace Conference
%% paper specifications by Wayne Blanding <wblanding@comcast.net> in
%% December 2012.
%%
%% Latest modifications made by Erica DeIonno in May 2018.
%%
%% For any suggestions or corrections to this class file, please contact
%% Erica DeIonno <erica.deionno@aero.org>. Do not change this file to use
%% it for the conference paper without first contacting Erica DeIonno.
%%
%% This class file is designed for a LaTeX2e program version. 
%%
%%
%%
%% WARNING: math font problems and workaround:
%%          Standard math commands like
%%		$\mathbf{N}(0,P(0))$
%%	    will not work properly. The reason is, that this is
%%	    a simple hack of an old LaTeX2.09 style files.
%%	    To get it running, I used the packages
%%		'rawfonts' and 'oldlfont'
%%	    They are responsible for the font problems.
%%	    This is a known feature, see
%%		http://www.tex.ac.uk/CTAN/latex/bugs.html
%%
%%	WORKAROUND:
%%	Use an additional pair of braces, like
%%	{\mathbf{y}}
%%	to introduce an additional level of scope.
%%	Examples:
%%		$\mathbf{N}(0,P(0))$ but ${\mathbf{N}}(0,P(0))$
%%		$\mathcal{N}(0,P(0))$ but ${\mathcal{N}}(0,P(0))$
%%		$\mathcal{G}_{t}$ but ${\mathcal{G}}_{t}$
%%		$\mathfrak{G}_t$ but ${\mathfrak{G}}_t$
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ProvidesClass{IEEEAerospaceCLS}[2012/12/10 IEEE Aerospace Conference 2012 document class] % WB
\typeout{Document Style `IEEEAerospaceCLS' <December 10, 2012>.}
\NeedsTeXFormat{LaTeX2e} % WB


\usepackage{times} % WB added--otherwise CMR is used
\usepackage{color} % WB added--for hidden symbol validation
\usepackage{rawfonts} %!PN
\usepackage{oldlfont} %!PN
\renewcommand{\normalsize}{\@normalsize} %!PN


\AtBeginDocument{ %RB vvv
  \usepackage{times}
  \setlength{\parskip}{1\baselineskip}
} %RB and %AT ^^^

%% define new needed boolean variables and initialize to default values
\newif\if@tmptwocolumn  \@tmptwocolumnfalse
\newif\if@draftversion  \@draftversionfalse
\newif\if@technote      \@technotefalse
\newif\if@appendix      \@appendixfalse

\def\@ptsize{10}
\@namedef{ds@9pt}{\def\@ptsize{9}}
\@namedef{ds@10pt}{\def\@ptsize{10}}
\@namedef{ds@11pt}{\def\@ptsize{11}}
\@namedef{ds@12pt}{\def\@ptsize{12}}
\@namedef{ds@twoside}{\@twosidetrue \@mparswitchtrue}
\@namedef{ds@draft}{\global\@draftversiontrue} % sets variable for draft
\@namedef{ds@technote}{\global\@technotetrue}  % sets variable for techn. note
%% the file twocolumn.sty is not read as it changes \textwidth.
%% Next line sets @tmptwocolumn to true which lets later execute \twocolumn
\@namedef{ds@twocolumn}{\global\@tmptwocolumntrue }
\@options

%%%%% \input IEEE\@ptsize.sty\relax  %%%% FONT DEFINITION BELOW (NO FILE READ)

%%% Formatting %AT
\usepackage[lmargin=0.75in,rmargin=0.75in,tmargin=0.70in,bmargin=0.80in]{geometry} %%AT
\geometry{twoside=true} % WB--deleted obsolete "twosideshift" parameter
\columnsep        .25in
\parindent          0in
\lineskip           1pt
\normallineskip     1pt
\def\baselinestretch{1}



%% FONT DEFINITION: avoids having to read in font files.
%%
%% Check if we have selected 10 points
\def\@tempa{10}\ifx\@ptsize\@tempa
\typeout{-- This is a 10 point document}
\def\@normalsize{\@setsize\normalsize{10pt}\xpt\@xpt
\abovedisplayskip 1em plus2pt minus5pt\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ plus3pt\belowdisplayshortskip .6em plus3pt minus3pt
\topsep \belowdisplayshortskip%!PN
}
\def\small{\@setsize\small{9pt}\ixpt\@ixpt}
\def\footnotesize{\@setsize\footnotesize{8pt}\viiipt\@viiipt}
\def\scriptsize{\@setsize\scriptsize{8pt}\viiipt\@viiipt}
\def\tiny{\@setsize\tiny{6pt}\vpt\@vpt}
\def\large{\@setsize\large{12pt}\xiipt\@xiipt}
\def\Large{\@setsize\Large{20pt}\xxpt\@xxpt}
\def\LARGE{\@setsize\LARGE{22pt}\xviipt\@xviipt}
\def\huge{\@setsize\huge{22pt}\xxpt\@xxpt}
\def\Huge{\@setsize\Huge{28pt}\xxvpt\@xxvpt}
\fi

%% Definitions for PARAGRAPHS
%%
\abovedisplayskip       0pt plus2pt minus5pt%
\belowdisplayskip       \abovedisplayskip
\abovedisplayshortskip  0pt plus3pt%
\belowdisplayshortskip  0pt plus3pt minus3pt%

%% Definitions for LISTS
%%

%% Change aspect of lists with
%% 1) \itemindent,  label indentation  wrt to left list margin
%% 2) \leftmargini, the indentation of the whole list (on left, first level)
\itemindent        -1em
\leftmargini        2em
\leftmarginii       1em
\leftmarginiii    1.5em
\leftmarginiv     1.5em
\leftmarginv      1.0em
\leftmarginvi     1.0em
\labelsep           5pt
\leftmargin\leftmargini
\labelwidth         \z@

\def\@listI{\leftmargin\leftmargini} \@listI
\def\@listi{\leftmargin\leftmargini  \topsep \z@ plus 1pt minus 1pt}
\def\@listii{\leftmargin\leftmarginii\labelwidth\leftmarginii
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listiii{\leftmargin\leftmarginiii\labelwidth\leftmarginiii
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listiv{\leftmargin\leftmarginiv\labelwidth\leftmarginiv
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listv{\leftmargin\leftmarginv\labelwidth\leftmarginv
    \advance\labelwidth-\labelsep    \topsep \z@}
\def\@listvi{\leftmargin\leftmarginvi\labelwidth\leftmarginvi
    \advance\labelwidth-\labelsep    \topsep \z@}

\def\labelenumi{\theenumi.}     \def\theenumi{\arabic{enumi}}
\def\labelenumii{(\theenumii)}  \def\theenumii{\alph{enumii}}
\def\labelenumiii{\theenumiii.} \def\theenumiii{\roman{enumiii}}
\def\labelenumiv{\theenumiv.}   \def\theenumiv{\Alph{enumiv}}
\def\p@enumii{\theenumi}
\def\p@enumiii{\theenumi(\theenumii)}
\def\p@enumiv{\p@enumiii\theenumiii}

\def\labelitemi{$\scriptstyle\bullet$}
\def\labelitemii{\bf --}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}

%% \itemindent is set to \z@ by list, so define new temporary variable
\newdimen\tmpitemindent
\def\verse{\let\\=\@centercr
    \list{}{\itemsep\z@ \itemindent -1.5em \listparindent \itemindent
    \rightmargin\leftmargin\advance\leftmargin 1.5em}\item[]}
\let\endverse\endlist
\def\quotation{\list{}{\listparindent 1.5em \itemindent\listparindent
    \rightmargin\leftmargin \parsep 0pt plus 1pt}\item[]}
\let\endquotation=\endlist
\def\quote{\list{}{\rightmargin\leftmargin}\item[]}
\let\endquote=\endlist
\def\@mklab#1{#1}
\def\description{\tmpitemindent\itemindent\list{}{\itemindent\tmpitemindent
    \labelwidth\z@\def\makelabel##1{\hspace\labelsep{##1}}}}
\def\enddescription{\endlist\par}
\def\enumerate{\tmpitemindent\itemindent\ifnum \@enumdepth >3 \@toodeep\else
    \advance\@enumdepth \@ne 
\edef\@enumctr{enum\romannumeral\the\@enumdepth}%
    \list{\csname label\@enumctr\endcsname}{\itemindent\tmpitemindent
    \usecounter{\@enumctr}\def\makelabel##1{\hspace\labelsep\hfil{##1}}}\fi}
\def\endenumerate{\endlist\par}
\def\itemize{\tmpitemindent\itemindent\ifnum \@itemdepth >3 \@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \list{\csname\@itemitem\endcsname}{\itemindent\tmpitemindent
    \def\makelabel##1{\hspace\labelsep\hfil{##1}}}\fi}
\def\enditemize{\endlist\par}

\newif\if@restonecol
\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
    \else \newpage \fi \thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi}

\arraycolsep     5pt
\tabcolsep       6pt
\arrayrulewidth .4pt
\doublerulesep   2pt
\tabbingsep\labelsep

%% FOOTNOTES
%
\skip\footins 10pt plus 4pt minus 2pt
\footnotesep 7pt
\footskip 7mm
\skip\@mpfootins = \skip\footins
\fboxsep = 3pt \fboxrule = .4pt
\long\def\@makefntext#1{$^{\@thefnmark}$#1} %WB--do not indent first line of footnote

\def\footnoterule{\kern-.4\p@
        %\hrule\@width 5pc
        \kern11\p@\kern-\footnotesep} %AT makes overline in footnotes - removed overline in footnote to conform to word template. ED 2015

 
%%%%%%%%
\setcounter{secnumdepth}{1}
\newcounter{section}
\newcounter{subsection}[section]
%\newcounter{subsubsection}[subsection]
%\newcounter{paragraph}[subsubsection]

\def\thesection{\arabic{section}} %RB and %AT
\def\thesubsection{\thesection-\Alph{subsection}}
\def\thesubsectiondis{} %RB and %AT
%\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
%\def\thesubsubsection{} %RB and %AT
%\def\thesubsubsectiondis{\Alph{subsection}.\arabic{subsubsection}}
%\def\thesubsubsectiondis{} %RB and %AT
%\def\theparagraph{\thesubsubsection.\alph{paragraph}}
%\def\theparagraphdis{\thesubsubsectiondis.\alph{paragraph}}

\def\theequation{\arabic{equation}}

%% LIST OF FIGURES AND TABLES
%
\def\@pnumwidth{1.55em}   % sets width of box in which the page number is set.
\def\@tocrmarg {2.55em}   % indentation of right margin for all but last line of multi-line entries
\def\@dotsep{1}           % separation between dots in mu (math units).

%% TABLE OF CONTENTS definition
%%
\setcounter{tocdepth}{1}  % only list section headings in TOC
\def\tableofcontents{\section*{ \bfseries{\scshape Table of Contents}}\@starttoc{toc}}
\def\l@section{\addvspace{-0.8em} \bfseries \scshape \@dottedtocline{1}{0em}{1.5em}}
\def\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\def\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}

% RS:  Redefine the \numberline macro to simply print its argument followed by a space.
% RS:  In this class file, \numberline is used exclusively inside \addcontentsline, which puts things into the table of contents.  By redefining it here, we're making it so that lines in the TOC start with the section number, followed by a space.  Without redefining the command, LaTeX does more advanced stuff to make the section number a fixed width part of the line.
\renewcommand{\numberline}[1]{#1 }

% RS: redefining contents line so it'll get rid of space between dots and page number.  It only works for single-digit page numbers.  Two-digit page numbers still have a little extra space.  It looks like there are one too few dots in the line.
\let \@conlin\contentsline
\renewcommand{\contentsline}[3]
{%
  \ifnum#3<10 % one-digit page number
    \def\@pnumwidth{0.5em}%
  \else % two-digit page number (page limit is 20 pages, so don't have to worry about three-or-more-digit numbers)
    \def\@pnumwidth{10pt}%
  \fi%
  \def\@tocrmarg{\@pnumwidth}%
  \@conlin{#1}{#2}{#3}%
}


%% Definitions for floats
%%
%% Normal Floats                      %% Double Column Floats
\floatsep 12pt plus 2pt minus 2pt     \dblfloatsep 12pt plus 2pt minus 2pt
\textfloatsep 20pt plus 2pt minus 4pt \dbltextfloatsep 20pt plus 2pt minus 4pt
\@fptop 0pt plus 1fil                 \@dblfptop 0pt plus 1fil
\@fpsep 8pt plus 2fil                 \@dblfpsep 8pt plus 2fil
\@fpbot 0pt plus 1fil                 \@dblfpbot 0pt plus 1fil
\def\topfraction{1.0}                 \def\dbltopfraction{1.0}
\def\floatpagefraction{0.8}           \def\dblfloatpagefraction{0.8}
\setcounter{topnumber}{2}             \setcounter{dbltopnumber}{4}
%
\intextsep 12pt plus 2pt minus 2pt
\setcounter{bottomnumber}{2}
\def\bottomfraction{.4}
\setcounter{totalnumber}{4}
\def\textfraction{.2}

%% Define ACKNOWLEDGMENTS        % added by WB
\def\acknowledgments{
  \section*{Acknowledgments}
  \addcontentsline{toc}{section}{Acknowledgments} %RS removed "\protect\numberline {}" before Acknowledgments so that Acknowledgments is not indented in the TOC
  \vspace{-3.5mm} %WB--adjust spacing
}

%% redefine CAPTION

\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname ext@#1\endcsname}{#1}{%
   \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}}
   \begingroup \@parboxrestore \normalsize
     \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}{#1}\par
   \endgroup}

\def\tablestring{table}
\def\figurestring{figure}

\newlength{\captionindent}
\long\def\@makecaption#1#2#3{
\ifx\figurestring#3
    \vskip 5pt
    \setbox\@tempboxa\hbox{\normalsize #1.~ #2}
    \ifdim \wd\@tempboxa >\hsize
       \setbox\@tempboxa\hbox{\normalsize #1.~ }
       \setlength\captionindent{\wd\@tempboxa} \divide\captionindent by 2
       \parbox[t]{\hsize}{\centering \normalsize   %AT replaced \small by \normalsize, ED added \centering so captions on more than one line remain centered 10/14
       \unhbox\@tempboxa#2}
    \else \hbox to\hsize{\normalsize\hfil\box\@tempboxa\hfil} %AT replaced 
\small 
\normalsize
    \fi
\else
    \begin{center} {\normalsize #1.~} {\normalsize \rm #2} \end{center} %AT replaced \small by \normalsize and \sc by \rm
    \vskip 8pt
\fi}


\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{{\bf Figure~\thefigure}} %AT
\def\figure{\@float{figure}}
\let\endfigure\end@float
\@namedef{figure*}{\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}

\newcounter{table}
\def\thetable{\@arabic\c@table} %AT
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{{\bf Table~\thetable}} %AT
\def\table{\@float{table}}
\let\endtable\end@float
\@namedef{table*}{\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}

\def\ps@headings{
    \def\@oddhead{\hbox{}\scriptsize\leftmark \hfil \thepage}
    \def\@evenhead{}
    \def\@oddfoot{}
    \def\@evenfoot{}}


\def\ps@titlepagestyle{
    \def\@oddhead{\hbox{}\scriptsize\leftmark \hfil \thepage}
    \def\@evenhead{\scriptsize\thepage \hfil \leftmark\hbox{}}
    \def\@oddfoot{}\def\@evenfoot{}
}
\def\rightmark{}\def\leftmark{}


%% Defines the command for putting the header. footernote{TEXT} is the same
%% as markboth{TEXT}{TEXT}. Here for compatibility with other style files.
% \def\markboth#1#2{\def\leftmark{\uppercase{#1}}\def\rightmark{\uppercase{#2}}}  % WB not needed
% \def\footernote#1{\markboth{#1}{#1}}                                            % WB not needed

\def\today{\ifcase\month\or
    January\or February\or March\or April\or May\or June\or
    July\or August\or September\or October\or November\or December\fi
    \space\number\day, \number\year}

%% BIBLIOGRAPHY
%
%% separate citations with "], ["
\def\@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
\def\@citea{}\@cite{\@for\@citeb:=#2\do {\@citea\def\@citea{],
[}\@ifundefined {b@\@citeb}{{\bf ?}\@warning {Citation `\@citeb' on
page \thepage \space undefined}}%
{\csname b@\@citeb\endcsname}}}{#1}}

\def\@cite#1#2{[{#1\if@tempswa, #2\fi}]}

%% Allow short (name-less) citations, when used in conjunction with a
%% bibliography style that creates labels like \citename{, }
\let\@internalcite\cite
\def\cite{\def\citename##1{##1}\@internalcite}
\def\shortcite{\def\citename##1{}\@internalcite}
\def\@biblabel#1{\def\citename##1{##1}[#1]\hfill}

\def\thebibliography#1{
    \section*{References}
    \addcontentsline{toc}{section}{References}  % added by WB for TOC, RS removed "\protect\numberline {}" before References so that References is not indented in the TOC
    \vspace{-3.5mm} %WB--adjust spacing
    \normalsize\list %AT \footnotesize replaced with \normalsize
    {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
    \advance\leftmargin\labelsep \itemsep 4pt plus 0.5pt %AT 0pt plus 0.5pt
    \usecounter{enumi}}
    \def\newblock{\hskip .11em plus .33em minus .07em}
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

%% appendix command for one single appendix
\def\appendix#1{\par
    \setcounter{section}{0}\setcounter{subsection}{0}
    \def\thesection{\Alph{section}} 
    \section*{Appendix#1}
    \addcontentsline{toc}{section}{Appendix#1}  % RS removed "\protect\numberline {}" before Appendix so that Appendix is not indented in the TOC
    \vspace{-3.5mm} %WB--adjust spacing
}
%% appendices command for multiple appendicies
\def\appendices{\par
\addcontentsline{toc}{section}{Appendices}
    \setcounter{section}{0}\setcounter{subsection}{0}
    \def\thesection{\Alph{section}} 
    \section*{Appendices}
%    \addcontentsline{toc}{section}{Appendices}  % RS commented out the whole line so that "Appendices" does not have an entry in the TOC.  The original line had "\protect\numberline {}" before Appendices.  RS removed that so that if Appendices is put back into the TOC, it will not be indented

    \@appendixtrue
    \vspace{-3.5mm} %WB--adjust spacing
}

%% Define thebiography        % added by WB
\def\thebiography{
  \section*{Biography\color{white}[} % WB--added hidden character for format checking
  \addcontentsline{toc}{section}{Biography} % RS removed "\protect\numberline {}" before Biography so that Biography is not indented in the TOC
  \vspace{-1mm} %WB--adjust spacing
}


%% MAKETITLE definition
%%
\def\maketitle{\par
  \begingroup
  \def\thefootnote{} % the \thanks{} mark type is empty
  \twocolumn[\@maketitle]  % WB--eliminated conditional checking for twocolumn
  \@thanks
  \endgroup
  \setcounter{footnote}{1}\let\maketitle\relax\let\@maketitle\relax
  \gdef\@thanks{}
  \gdef\@author{}
  \gdef\@title{}\let\thanks\relax
}

\def\@maketitle{
    \newpage
    \begin{center}
    \renewcommand{\baselinestretch}{1.2}   %WB--larger separation between lines
    \vskip0.2em{\Large\bf\@title\par}\vskip1.0em%  %AT 
    \renewcommand{\baselinestretch}{1}     %WB--restore original separation
    \lineskip.5em\normalsize
    {\bf\begin{tabular}[t]{c}% WB--fixes bug in which multiple authors could not be listed on same line
    \@author
    \end{tabular}\par} % AT \large
    \end{center}\par\vskip 1.4em
}


%% ABSTRACT definition
\def\abstract{\small\bf\subsubsection{\bfseries Abstract}}   % WB--removed conditional statement referring to 2 column mode; converted to bold font 6/2011
\def\endabstract{\vspace{0.6em}\par\normalsize\rm}


%% KEYWORDS definition    % WB--removed; keywords not part of standard paper formatting
%\def\keywords{\vspace{-.3em}
%    \if@twocolumn
%      \subsubsection{Keywords}%RB
%    \else
%      \begin{center}\small\bf Keywords\end{center}\quotation\small
%    \fi}
%\def\endkeywords{\vspace{0.6em}\par\if@twocolumn\else\endquotation\fi
%    \normalsize\rm}

%% FIX numbering of sections.
\gdef\@punct{\/---\hspace{-3pt}}  % Punctuation after run-in section heading %WB
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
     \def\@svsec{}
  \else
     \refstepcounter{#1}\edef\@svsec{%
     \ifnum #2=1{\if@appendix \Alph{#1}.\hskip.25em\else\arabic{#1}.\hskip.25em\fi}\fi %RB
   } %RB
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
     \begingroup #6\relax
       \@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M #8\par}
     \endgroup
     \csname #1mark\endcsname{#7}
     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth\else
     	  \protect\numberline{\csname the#1\endcsname.}\fi#7} % RS added the "." after \endcsname so section numbers would end with dots in the TOC
  \else
     \def\@svsechd{#6\hskip #3\@svsec #8\@punct\csname #1mark\endcsname{#7}
     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth \else
     	  \protect\numberline{\csname the#1\endcsname.}\fi#7}} % RS added the "." after \endcsname so section numbers would end with dots in the TOC
  \fi
  \@xsect{#5}
  \ifnum #2<3{\vspace{-\baselineskip}}\fi %RB
  } %RB

\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
     \begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M #5\par}\endgroup
  \else \def\@svsechd{#4\hskip #1\relax #5\@punct}\fi
  \@xsect{#3}}

%% SECTIONS
%%
\def\section{
\@startsection{section}{1}{\z@}{1.2\baselineskip}{0.5\baselineskip}{\centering\large\bfseries\scshape}}   % WB add bold face sm cap
\def\subsection{\@startsection{subsection}{2}{\z@}{0mm}{.5\baselineskip}{\it}}
\def\subsubsection{\@startsection{subsubsection}{3}{0mm}{-0mm}{0mm}{\it}}
\def\paragraph{\@startsection{paragraph}{4}{\z@}{1.0ex plus .5ex minus .2ex}{0ex}{\rm}}

%% ENVIRONMENTS
\def\QED{\mbox{\rule[0pt]{1.5ex}{1.5ex}}}
\def\proof{\noindent\hspace{2em}{\it Proof: }}
\def\endproof{\hspace*{\fill}~\QED\par\endtrivlist\unskip}
\def\@begintheorem#1#2{\tmpitemindent\itemindent\topsep 0pt\rm\trivlist
    \item[\hskip \labelsep{\indent\it #1\ #2:}]\itemindent\tmpitemindent}
\def\@opargbegintheorem#1#2#3{\tmpitemindent\itemindent\topsep 0pt\rm 
\trivlist
    \item[\hskip\labelsep{\indent\it #1\ #2\ 
\rm(#3)}]\itemindent\tmpitemindent}
\def\@endtheorem{\endtrivlist\unskip}

%% sets up default PAGESTYLE
\ps@headings
\pagenumbering{arabic}
\twocolumn \sloppy \flushbottom % the default is 2-column. WB 10/18/2010
\hyphenpenalty=750              % added to reduce hyphenation of text WB 10/18/2010

%% Definition for Big letter at the beginning of a paragraph
%%
\def\PARstart#1#2{\begingroup\def\par{\endgraf\endgroup\lineskiplimit=0pt}
    \setbox2=\hbox{\uppercase{#2} }\newdimen\tmpht \tmpht \ht2
    \advance\tmpht by \baselineskip\font\hhuge=cmr10 at \tmpht
    \setbox1=\hbox{{\hhuge #1}}
    \count7=\tmpht \count8=\ht1\divide\count8 by 1000 \divide\count7 by \count8
    \tmpht=.001\tmpht\multiply\tmpht by \count7\font\hhuge=cmr10 at \tmpht
    \setbox1=\hbox{{\hhuge #1}} \noindent \hangindent1.05\wd1
    \hangafter=-2 {\hskip-\hangindent \lower1\ht1\hbox{\raise1.0\ht2\copy1}%
    \kern-0\wd1}\copy2\lineskiplimit=-1000pt}


%% Allows to enter BIOGRAPHY leaving place for picture (adapts to font size)
%%
\newenvironment{biography}[1]{%
\normalsize\unitlength 1mm\bigskip\parskip=0pt\par %WB--Adjust spacing
\rule{0pt}{39mm}\vspace{-39mm}\par%   garantees correct page breaking
\noindent\setbox0\hbox{\framebox(25,32){}}%   box containing the frame
\ht0=39mm\count10=\ht0\divide\count10 by\baselineskip%  calculates lines
\global\hangindent29mm\global\hangafter-\count10%
\hskip-28.5mm\setbox0\hbox to 28.5mm {\raise-30.5mm\box0\hss}%
\dp0=0mm\ht0=0mm\box0\noindent{\textbf{#1}}}{ %AT \bf#1>>\textbf{#1},\rm>>\em %WB 
\par\rm\normalsize}


%% Allows to enter BIOGRAPHY with picture (adapts to font size) %AT
%%
\newenvironment{biographywithpic}[2]{%
\normalsize\unitlength 1mm\bigskip\parskip=0pt\par %WB--adjust spacing
\rule{0pt}{39mm}\vspace{-39mm}\par%   garantees correct page breaking
\noindent\setbox0\hbox{\includegraphics[width=25mm,height=32mm]{#2}} %PW: modern graphics
\ht0=37mm\count10=\ht0\divide\count10 by\baselineskip%  calculates lines
\global\hangindent29mm\global\hangafter-\count10%
\hskip-28.5mm\setbox0\hbox to 28.5mm {\raise-30.5mm\box0\hss}%
\dp0=0mm\ht0=0mm\box0\noindent\it{\textbf{#1}}}{ %AT \bf#1>>\textbf{#1},\rm>>\em %WB 
\par\rm\normalsize}

%% Allows a BIOGRAPHY entry with a JPEG picture (adapts to font size) %PD
%% Must \usepackage[pdftex,final]{graphicx}
%%
\newenvironment{biographywithjpg}[2]{ %
\normalsize\unitlength 1mm\bigskip\parskip=0pt\par %WB--adjust spacing
\rule{0pt}{39mm}\vspace{-39mm}\par%   garantees correct page breaking
\noindent\setbox0\hbox{\includegraphics[height=32mm]{\fighome #2}} %AT \framebox is replaced with \psfig{}
\ht0=39mm\count10=\ht0\divide\count10 by\baselineskip %  calculates lines
\global\hangindent29mm\global\hangafter-\count10 %
\hskip-28.5mm\setbox0\hbox to 28.5mm {\raise-30.5mm\box0\hss} %
\dp0=0mm\ht0=0mm\box0\noindent{\textbf{#1}}}{ %AT \bf#1>>\textbf{#1},\rm>>\em %WB
\par\rm\normalsize}

\endinput %!PN