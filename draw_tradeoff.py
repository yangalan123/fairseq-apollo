import matplotlib.pyplot as plt
import numpy as np

#set axis limits of plot (x=0 to 20, y=0 to 20)
plt.axis([-1, 16, 43, 60])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16], [0, 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([45, 47, 49, 51, 53, 55, 57, 59])

# plt.xticks(np.arange(40, 60, 10.0))

plt.gca().set_aspect('equal', adjustable='box')
#define circles
# c1=plt.Circle((5, 5), radius=1)
# c2=plt.Circle((10, 10), radius=2)
# c3=plt.Circle((15, 13), radius=3)
Transformer = 54.02
Local_Attention = 44.22
Sparse_Trans = 49.15
Longformer = 52.6
Linformer = 51.14
Reformer = 49.99
Sinkhorn_Trans = 50.89
Synthesizer = 52.43
BigBird = 53.94
Linear_Trans = 49.92
Performer = 50.81
luna_16 = 55.93
luna_128 = 56.65
luna_256 = 56.67

Transformer_speed = 1
Local_Attention_speed = 5.285714286
Linformer_speed = 5.5
Reformer_speed = 0.7857142857
Sinkhorn_Trans_speed = 3.785714286
Synthesizer_speed = 1.357142857
BigBird_speed = 1.071428571
Linear_Trans_speed = 5.571428571
Performer_speed = 5.714285714
luna_16_speed = 5.788925698
luna_128_speed = 4.828442972
luna_256_speed = 4.305253195

Transformer_memory = 1.5
Local_Attention_memory = 0.1445147679*1.5
Linformer_memory = 0.1044303797*1.5
Reformer_memory = 0.2405063291*1.5
Sinkhorn_Trans_memory = 0.1561181435*1.5
Synthesizer_memory = 0.7373417722*1.5
BigBird_memory = 0.3037974684*1.5
Linear_Trans_memory = 0.108649789*1.5
Performer_memory = 0.111814346*1.5
luna_16_memory = 0.083267055*1.5
luna_128_memory = 0.1107146858*1.5
luna_256_memory = 0.1465721967*1.5

c1=plt.Circle((5, 5), radius=1)

#add circles to plot
plt.gca().add_artist(plt.Circle((Transformer_speed*2, Transformer), radius=Transformer_memory, alpha=0.5, linewidth=1.5, ec='black', color='violet'))
plt.text(Transformer_speed*2, Transformer+1.5, 'Transformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Local_Attention_speed*2, Local_Attention), radius=Local_Attention_memory, alpha=0.5, linewidth=1.5, ec='black', color='cyan'))
plt.text(Local_Attention_speed*2, Local_Attention+Local_Attention_memory, 'Local Attention', fontsize=12)
plt.gca().add_artist(plt.Circle((Linformer_speed*2, Linformer), radius=Linformer_memory, alpha=0.5, linewidth=1.5, ec='black', color='tomato'))
plt.text(Linformer_speed*2-6*Linformer_memory, Linformer+2*Linformer_memory, 'Linformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Reformer_speed*2, Reformer), radius=Reformer_memory, alpha=0.5, linewidth=1.5, ec='black', color='orange'))
plt.text(Reformer_speed*2, Reformer+Reformer_memory, 'Reformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Sinkhorn_Trans_speed*2, Sinkhorn_Trans), radius=Sinkhorn_Trans_memory, alpha=0.5, linewidth=1.5, ec='black', color='lime'))
plt.text(Sinkhorn_Trans_speed*2, Sinkhorn_Trans+Sinkhorn_Trans_memory, 'Sinkhorn', fontsize=12)
plt.gca().add_artist(plt.Circle((Synthesizer_speed*2, Synthesizer), radius=Synthesizer_memory, alpha=0.5, linewidth=1.5, ec='black', color='blueviolet'))
plt.text(Synthesizer_speed*2, Synthesizer, 'Synthesizer', fontsize=12)
plt.gca().add_artist(plt.Circle((BigBird_speed*2, BigBird), radius=BigBird_memory, alpha=0.5, linewidth=1.5, ec='black', color='royalblue'))
plt.text(BigBird_speed*2+BigBird_memory, BigBird, 'BigBird', fontsize=12)
plt.gca().add_artist(plt.Circle((Linear_Trans_speed*2, Linear_Trans), radius=Linear_Trans_memory, alpha=0.5, linewidth=1.5, ec='black', color='chocolate'))
plt.text(Linear_Trans_speed*2-10*Linear_Trans_memory, Linear_Trans-5*Linear_Trans_memory, 'Linear Transformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Performer_speed*2, Performer), radius=Performer_memory, alpha=0.5, linewidth=1.5, ec='black', color='olive'))
plt.text(Performer_speed*2+2*Performer_memory, Performer, 'Performer', fontsize=12)
plt.gca().add_artist(plt.Circle((luna_16_speed*2, luna_16), radius=luna_16_memory, alpha=0.8, linewidth=1.5, ec='black', color='red'))
plt.text(luna_16_speed*2 + 2*luna_16_memory, luna_16, 'Luna-16', fontsize=12)
plt.gca().add_artist(plt.Circle((luna_128_speed*2, luna_128), radius=luna_128_memory, alpha=0.8, linewidth=1.5, ec='black', color='red'))
plt.text(luna_128_speed*2+ 2*luna_128_memory, luna_128, 'Luna-128', fontsize=12)
plt.gca().add_artist(plt.Circle((luna_256_speed*2, luna_256), radius=luna_256_memory, alpha=0.8, linewidth=1.5, ec='black', color='red'))
plt.text(luna_256_speed*2-6*luna_256_memory, luna_256+2*luna_256_memory, 'Luna-256', fontsize=12)
plt.gca().add_artist(plt.Rectangle((7, 55.5), 7.5, 2.1, alpha=1, ec='red', facecolor='none', linestyle="dotted"))
plt.text(7, 58, 'Our Proposed Work', fontsize=14, weight='bold')
plt.grid()
plt.xlabel("Relative Speed Comparision", fontsize=15)
plt.ylabel("Avg. LRA Score (w/o Retrieval)", fontsize=15)
plt.savefig('overall_tradeoff.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
  


