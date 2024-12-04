### Regra 1

O nome das variáveis privadas deve iniciar com underscore.

### Regra 2

O nome das constantes deve utilizar a capitalização Pascal Casing.

### Regra 3

Não usar a palavra reservada `this` para acesso os elementos da própria classe. 

### Regra 4

Constantes, quando existem, devem ser declaradas em sequência.
Variáveis, quando existem, devem ser declaradas em sequência.
Propriedades, quando existem, devem ser declaradas em sequência.
Construtores, quando existem, devem ser declarados em sequência.

### Regra 5

Quando existirem, os métodos devem aparecer nesta ordem: Primeiro, métodos public. Depois, métodos protected. Depois, métodos privados.

### Regra 6

Os únicos sufixos válidos para classes estão na seguinte tabela:

| **Camada** | **Sufixo** |
|--|--|
| Controladores | Controller |
| Filtros | Filter |
| Cache | Cache |
| Processamento de fila | Processor |
| Regra de negócio | Manager |
| Acesso a dados | Repository |
| Transferência de dados/objetos | Dto |
| Objeto de filtro recebido pela controller | Filter |
| Exceções customizadas | Exception |
| Anotações | Attribute |
| Classes para extensão de funcionalidades | Extension |
| Resiliência | Resilience |

### Regra 7

Classes Dto (Data Transfer Objects) não devem conter nenhum método.

