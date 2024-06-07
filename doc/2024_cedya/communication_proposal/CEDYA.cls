%
%  ╔══════════════════════════════════════════════╗
%  ║ © 2022                                       ║
%  ║ Eduardo Martínez Fernández                   ║
%  ║ emf@unizar.es                                ║
%  ║ Departamento de Matemática Aplicada          ║
%  ║ Facultad de Ciencias                         ║
%  ║ Universidad de Zaragoza                      ║
%  ╚══════════════════════════════════════════════╝
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CEDYA}[CEDYA 2024 LaTeX class]

\LoadClass[a4paper,11pt]{article}
\RequirePackage{amsmath, amsthm, amscd, amssymb, amsfonts, graphicx, hyperref}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}

\RequirePackage{fancyhdr, color}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\lhead{\textcolor{blue}{\textsc{CEDYA 2024 - \proposal}}}
\rhead{\textcolor{blue}{Section: \MakeUppercase{\SECTION}}}

% proposals
\DeclareOption{minisymposium}{%
	\def\proposal{Minisymposium proposal}%
	\def\other#1#2{\item \href{#2}{#1}}%
}
\DeclareOption{communication}{%
	\def\proposal{Communication proposal}%
}
\DeclareOption{poster}{%
	\def\proposal{Poster proposal}%
}
% talks
\DeclareOption{plenary-talk}{%
	\lhead{\textcolor{blue}{\textsc{CEDYA 2024 - Plenary talk}}}%
	\rhead{}%
}
\DeclareOption{minisymposium-talk}{%
	\lhead{\textcolor{blue}{\textsc{Minisymposium:\enspace}\ignorespaces\MINISYMPOSIUM}}%
	\rhead{}%
}
%
\ProcessOptions\relax

%----------------------------------------------------------------
%-----------
% CEDYA sections DS-ODE PDE NAS NLA OC-IP AM ME OTHER
% PDE	  : Partial Differential Equations
% DS-ODE  : Dynamical Systems-Ordinary Differential Equations
% NAS     : Numerical Analysis and Simulation
% OC-IP   : Control and Optimization-Inverse problems
% NLA     : Numerical Liner Algebra
% AM	  : Applied Mathematics
% ME	  : Mathematical Education
% OTHER   : Other thematic not included above (Scientific calculus, Approximation theory, Discrete mathematics, ...)
%-----------
% CEDYAsection
\newcommand{\CEDYAsection}[1]{\gdef\SECTION{#1}}

% title of the proposal
\renewcommand{\title}[1]{\noindent\textbf{\ignorespaces#1}\par\bigskip}
% set of persons (itemize)
\newenvironment{authors}{\par\noindent\textbf{Authors:}\itemize}{\enditemize}
\newenvironment{organizers}{\par\noindent\textbf{Organizers:}\itemize}{\enditemize}
\newenvironment{speakers}{\medskip\par\noindent\textbf{Speakers:}\itemize}{\enditemize}
% data of person
\def\person{\@ifstar\@person\@@person}
\def\@person#1#2#3#4{\item \underline{#2 #1}, #3 (\texttt{#4})}
\def\@@person#1#2#3#4#5{\item #2 #1, #3 (\texttt{#4}) #5}
\let\speaker\person
\let\organizer\person
\let\author\person
% abstract/description of the proposal
\renewenvironment{abstract}{\par\vspace{12pt}\noindent\textbf{Abstract:}\enspace\ignorespaces}{\par}
% reformat slightly thebibliography
\renewenvironment{thebibliography}[1]
     {\par\medskip\par\noindent\textbf{References:}
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

%----------------------------------------------------------------
% 
\newcommand{\minisymposium}[2][]{\gdef\ID{#1}\gdef\MINISYMPOSIUM{#2}}
